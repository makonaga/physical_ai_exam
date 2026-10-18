# 段階1 実行手順（SageMaker Studio）

25mm 角の立方体 1 個を鍋に入れるタスクで、データ生成 → ACT 学習 → 評価を一周する。

## 全体の流れ

| 手順 | インスタンス | 内容 | 所要時間の目安 |
|---|---|---|---|
| A | `ml.c5.9xlarge`（CPU 36 vCPU） | 学習データの生成（成功 300 エピソード） | 5 分程度 |
| B | `ml.g5.xlarge`（GPU） | ACT の学習（40,000 ステップ） | 1〜2 時間（初回に実測する） |
| C | `ml.c5.9xlarge`（CPU） | 学習済みポリシーの評価（50 エピソード） | 数分 |

シミュレーションの描画は CPU で行うため、A と C は vCPU の多い CPU インスタンスで並列に動かす。
B の学習だけ GPU を使う。データと学習結果はリポジトリのフォルダ（`~/user-default-efs/physical_ai_exam`）
の下に保存されるので、インスタンスを切り替えても残る。

## 共通：インスタンスを切り替えて起動したら毎回やること

1. SageMaker Studio のスペース一覧でスペースを停止し、インスタンスタイプを選び直して起動する。
2. JupyterLab でターミナルを開き（File → New → Terminal）、次の 3 行を実行する。

   ```bash
   cd ~/user-default-efs/physical_ai_exam
   git pull
   bash scripts/setup_sagemaker.sh
   ```

   最後に `完了` と出れば準備は終わり。GPU なしのインスタンスでは `CUDA 利用可能: False` と出るが、正常。

以下のコマンドは、どれも `~/user-default-efs/physical_ai_exam` で実行する。
処理は裏で動くので、ブラウザを閉じても止まらない（スペースを停止すると止まる）。

## A. 学習データの生成（ml.c5.9xlarge）

```bash
bash scripts/generate_dataset.sh pot_cube_v1 300
```

- 表示された `tail -f ...` のコマンドで進み具合を見られる（見るのをやめるときは Ctrl+C。処理は止まらない）。
- `[worker 05] 3/9 ...` のような行が 36 プロセス分流れ、最後に `完了: .../data/pot_cube_v1` と出たら終わり。
- 結果の確認:

  ```bash
  head -20 data/pot_cube_v1/generation_stats.json
  ```

  `episodes`（集めた成功エピソード数）と `expert_success_rate`（エキスパートの成功率）を見る。
- 生成したデータを動画で見るには、`bash scripts/export_dataset_videos.sh pot_cube_v1 6` で再生用の動画を作り、
  `notebooks/view_videos.ipynb` の 1 行目を `FOLDER = "outputs/dataset_videos/pot_cube_v1"` にして
  Run → Run All Cells を押す（データセットの中の動画は全エピソードが 1 本につながった AV1 形式で、
  そのままではブラウザで再生できないことがある）。

## B. ACT の学習（ml.g5.xlarge）

```bash
bash scripts/train_act.sh pot_cube_v1 act_pot_cube_v1
```

- 初回は ResNet18 の学習済み重みをダウンロードするので、始まるまで少しかかる。
- ログには 200 ステップごとに `step:200 ... loss:...` のような行が出る。`loss` が下がっていけば順調。
  始まって数分たったら一度ログを見て、エラーで止まっていないか確認する。
- `End of training` と出たら終わり。学習済みモデルは `outputs/train/act_pot_cube_v1/checkpoints/` に
  10,000 ステップごとに保存される（`last` が最新）。
- 学習中はスペースを停止しないこと。途中で止まった場合は、train_act.sh の先頭のコメントにある方法で再開できる。

## C. 評価（ml.c5.9xlarge）

```bash
bash scripts/evaluate.sh act_pot_cube_v1 50
```

- 終わったら結果を確認する:

  ```bash
  cat outputs/eval/act_pot_cube_v1/summary.json
  ```

  `success_rate` が成功率、`outcomes` が結果の内訳（`success` 成功 / `not_grasped` 掴めなかった /
  `missed_pot` 持ち上げたが鍋に入らなかった）。
- 先頭 6 エピソードの動画が `outputs/eval/act_pot_cube_v1/videos/` に保存される
  （ファイル名の末尾が結果。左が俯瞰カメラ、右が手首カメラ）。
- CPU インスタンスに切り替えずに GPU インスタンスのまま評価もできる（4 並列なので 10 分程度かかる）。

## 結果の共有

次の 3 つを共有すると、次の改善（段階2 以降）につなげられる。

1. `data/pot_cube_v1/generation_stats.json` の先頭部分（A）
2. 学習ログの最後の数行（`tail -5 logs/train_act_pot_cube_v1_*.log`）（B）
3. `outputs/eval/act_pot_cube_v1/summary.json`（C）

## 設計メモ

- **タスク**: 小片（25mm 角）はロボット前方の半径 7cm の円内（中心 x=0.21m）にランダムな向きで置く。
  鍋は (0.20, 0.20) に固定（内径 11cm、高さ 5cm）。成功は「小片の中心が鍋の内側・縁より下にある」。
- **エキスパート**: 真上把持の IK で経由点を作り、関節空間で補間する（接近 → 下降 → 把持 → 持ち上げ →
  鍋の上へ運搬 → 開放 → 開始姿勢へ戻る）。区間の時間・高さ・落とす位置はエピソードごとに少し揺らす。
  開発環境での 300 回の試行では成功率 99.3%。
- **データ**: 25Hz（シミュレーションは 50Hz で、同じ目標を 2 ステップ保持）。1 エピソード 155〜190 フレーム。
  入力は俯瞰カメラ・手首カメラ（224×224）と 6 関節の角度、出力は 6 関節の目標角度。
  角度の単位は LeRobot の SO-101 実機と同じ（腕は度、グリッパは 0〜100）。
- **生成の流れ**: 各プロセスがエキスパートを描画なしで実行し、成功したエピソードだけを状態列から描画し直して
  LeRobot v3 形式で保存する。最後にプロセスごとのデータを 1 つに統合する。
- **ACT**: 行動チャンク 50（2 秒分）、バッチ 16、40,000 ステップ。その他は LeRobot の既定値。
- **評価**: 学習に使っていない seed（100000〜）で配置した小片に対して、最大 10.4 秒動かす。
