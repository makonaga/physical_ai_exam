# VLA（SmolVLA）の学習・評価の手順

段階2の学習データ（`pot_multi_v1`: 散らばった食材片から一番近い片を鍋に入れる、成功 1,000 エピソード）で、
ACT の代わりに VLA（Vision-Language-Action モデル）を学習し、同じ条件で評価して ACT と比べる。

## 使う VLA: SmolVLA

| モデル | 大きさ | このプロジェクトの GPU（A10G 24GB）で学習 | 備考 |
|---|---|---|---|
| **SmolVLA**（`lerobot/smolvla_base`） | 約 4.5 億パラメータ | できる | SO-100/101（このロボット）のデータで事前学習済み。LeRobot 公式 |
| π0 / π0.5 | 約 30 億 | 全体の学習はメモリが足りない | より大きな GPU が要る |
| GR00T N1.7 | 約 30 億 | 同上 | 同上 |

SmolVLA を選んだ理由は、同じ SO-101 のデータで事前学習されていること、今の GPU で学習できること、
指示文（言葉）を入力に取れることの 3 つ。指示文は、段階3で「じゃがいもを入れて」のように食材を言葉で選ぶときに使える。

SmolVLA の中身と、学習するところ：

- 入力: 俯瞰・手首カメラの画像（512×512 に拡大）、6 関節の角度、指示文
  （このデータでは "Put the food piece closest to the robot into the pot."）
- 画像と指示文は、言語・画像モデル SmolVLM2（の前半 16 層）で読む。この部分は固定して学習しない
- 行動を出す部分（action expert、約 1 億パラメータ）だけを学習する。50 ステップ（2 秒）分の行動を、
  ノイズから 10 回かけて少しずつ整える方法（flow matching）で出す
- 事前学習済みモデルとこのデータではカメラ名が違う（overhead / wrist）ので、学習スクリプトが自動で対応付ける

## 全体の流れ

| 手順 | インスタンス | 内容 | 所要時間の目安 |
|---|---|---|---|
| A | `ml.g5.xlarge` | 準備（追加パッケージのインストール） | 数分 |
| B | `ml.g5.xlarge` | 試運転（200 ステップの学習と 2 エピソードの評価）で、動くことと速さを確かめる | 20〜30 分 |
| C | `ml.g5.12xlarge`（または `ml.g5.xlarge`） | 本番の学習（合計バッチ 64 × 20,000 ステップ） | 2〜2.5 時間（`ml.g5.xlarge` なら約 7 時間） |
| D | `ml.g5.12xlarge` のまま（または `ml.c5.9xlarge`） | 評価（100 エピソード × 2）と ACT との比較 | 1 種類あたり 10 分程度 |

以下のコマンドは、すべて `~/user-default-efs/physical_ai_exam` で実行する。

## A. 準備（ml.g5.xlarge）

インスタンスを `ml.g5.xlarge` にして起動し、ターミナルで次を実行する。

```bash
cd ~/user-default-efs/physical_ai_exam
git pull
bash scripts/setup_sagemaker.sh
```

今回は SmolVLA 用のパッケージ（transformers など）が追加で入るので、いつもより 1〜2 分長くかかる。
最後に `CUDA 利用可能: True（NVIDIA A10G）` と出れば準備完了。

## B. 試運転（ml.g5.xlarge）

いきなり数時間の学習を始めると、途中で失敗したときの損が大きいので、まず短く動かして確かめる。

### B-1. 200 ステップだけ学習する

```bash
bash scripts/train_smolvla.sh pot_multi_v1 smolvla_trial --steps=200 --save_freq=200 --log_freq=20
```

- 最初に、事前学習済みモデルのカメラ名と、このデータのカメラ名の対応が表示される
  （例: `observation.images.overhead → observation.image`）。
- そのあと学習が裏で始まる。初回は事前学習済みモデル（数 GB）のダウンロードがあるので、
  学習が始まるまで数分かかる。
- 進み具合: `tail -f logs/train_smolvla_trial_*.log`（見るのをやめるときは Ctrl+C）。
  `step:20 ... loss:... updt_s:... mem_gb:...` のような行が 20 ステップごとに出る。
  `updt_s` は 1 ステップの時間 [秒]、`mem_gb` は GPU のメモリ使用量 [GB]。
- `End of training` と出たら終わり。

### B-2. 試運転のモデルで評価が動くか確かめる

```bash
bash scripts/evaluate.sh smolvla_trial 2 --task multi --max-steps 100 --videos 1 --workers 2 --torch-threads 2
```

200 ステップしか学習していないので成功はしない（`success_rate` は 0 でよい）。
`tail -f logs/eval_smolvla_trial_*.log` で見て、最後に `完了` と出ればよい。

### B-3. 結果を確認して共有する

```bash
grep "step:200" logs/train_smolvla_trial_*.log | tail -n 1
tail -n 3 logs/eval_smolvla_trial_*.log
```

この 2 つの出力を共有する。`updt_s`（1 ステップの時間）と `mem_gb`（GPU メモリ）から、本番の学習の設定と
時間を見積もる（実際の試運転: バッチ 32 で 0.63 秒/ステップ、7.7GB）。
確認できたら、試運転の出力は消してよい。

```bash
rm -rf outputs/train/smolvla_trial outputs/eval/smolvla_trial
```

## C. 本番の学習（ml.g5.12xlarge 推奨）

GPU が 4 個ある `ml.g5.12xlarge`（A10G × 4、48 vCPU、メモリ 192GB）を使うと、4 個の GPU で並列に学習して
2〜2.5 時間で終わる（`ml.g5.xlarge` だと約 7 時間）。1 時間あたりの料金は約 5 倍だが、時間が約 1/3 になるので、
合計の料金は 1.5 倍程度。インスタンスの一覧に `ml.g5.12xlarge` が出ない場合は、AWS の Service Quotas で
SageMaker の「Studio JupyterLab Apps running on ml.g5.12xlarge instance」の上限を 1 以上に申請する。

GPU インスタンスは混み合っていて起動できない（`EC2InsufficientCapacityError`）ことがある。その場合は次から選ぶ
（学習時間は試運転の A10G の速さと、GPU の一般的な性能比から見積もったもの）。

| インスタンス | GPU | 学習時間の見込み | 備考 |
|---|---|---|---|
| `ml.g5.12xlarge` | A10G 24GB × 4 | 2〜2.5 時間 | |
| `ml.g6e.xlarge` | L40S 48GB × 1 | 3〜4 時間 | GPU 1 個なので仕組みが試運転と同じ。料金は `ml.g5.xlarge` の約 2 倍 |
| `ml.g6.12xlarge` | L4 24GB × 4 | 3〜4 時間 | L4 は 1 個あたり A10G よりやや遅い |
| `ml.g5.xlarge` | A10G 24GB × 1 | 約 7 時間 | 試運転と同じ |
| `ml.g6.xlarge` | L4 24GB × 1 | 7 時間以上 | `ml.g5.xlarge` より速くならないので使わない |

インスタンスを切り替えたら、`cd` / `git pull` / `setup_sagemaker.sh` の 3 行を実行してから次を実行する
（どのインスタンスでも同じコマンド。GPU の数はスクリプトが数える）。

```bash
bash scripts/train_smolvla.sh pot_multi_v1 smolvla_pot_multi_v1
```

- スクリプトが GPU の数を数えて、全部の GPU で学習する。最初に
  `GPU 4 個で学習します（1 GPU あたりのバッチ 16、合計 64）。` のように表示される。
- 合計バッチ 64 × 20,000 ステップは、lerobot の SmolVLA の手順と同じ設定（使うサンプル数の合計は ACT の学習と同じ 128 万）。
  GPU の数を変えても、合計バッチとステップ数は同じなので、同じ学習になる。
- 試運転（A10G 1 個、バッチ 32）では 1 ステップ 0.63 秒（1 秒あたり 50 サンプル）、GPU メモリ 7.7GB だった。
- 5,000 ステップごとにモデルが保存される（`outputs/train/smolvla_pot_multi_v1/checkpoints/`）。
- 進み具合: `tail -n 3 logs/train_smolvla_pot_multi_v1_*.log`。始まって 5 分ほどたったら一度見て、
  `step:...` の行が出ていること（エラーで止まっていないこと）を確認する。
- 学習中はスペースを Stop しないこと（Stop すると学習も止まる）。ブラウザは閉じてよい。
- SageMaker に「操作がないとスペースを自動停止する」設定（アイドルシャットダウン）があると、ブラウザを閉じて
  しばらくすると学習ごと止まることがある。念のため `notebooks/watch_training.ipynb` を開き、1 行目を
  `RUN = "smolvla_pot_multi_v1"` にして Run All Cells しておく。学習が終わるまでセルが「実行中」になるので、
  自動停止の対象になりにくい（ブラウザを閉じてもセルは動き続ける）。進み具合も 1 分ごとに表示される。
- 途中で止まったときは、`scripts/train_smolvla.sh` の先頭のコメントにある方法で、最後に保存した
  チェックポイント（5,000 ステップごと）から再開できる。
- 手元のパソコンから進み具合を見たいときは、学習を始めたあと SageMaker のターミナルで
  `bash scripts/sync_progress_s3.sh smolvla_pot_multi_v1 s3://<バケット名>/<フォルダ>` を実行しておく。
  5 分ごとに `<フォルダ>/smolvla_pot_multi_v1_progress.txt` に最新の進み具合が書かれるので、手元では
  `aws s3 cp s3://<バケット名>/<フォルダ>/smolvla_pot_multi_v1_progress.txt -` で見られる
  （手元に AWS CLI と、そのバケットを読める認証情報が必要）。

## D. 評価と ACT との比較（ml.g5.12xlarge のまま、または ml.c5.9xlarge）

評価は CPU で行う（シミュレーションの描画も CPU）。`ml.g5.12xlarge` は vCPU が 48 あるので、学習が終わったら
そのまま評価してよい（`ml.g5.xlarge` で学習した場合は、vCPU が 4 しかないので `ml.c5.9xlarge` に切り替え、
`cd` / `git pull` / `setup_sagemaker.sh` の 3 行を実行してから評価する）。
次を 1 つずつ実行する（前のログに `完了` が出てから次へ）。

```bash
# 1. 初めて見る形（ACT の act_pot_multi_v1 と同じ 100 の配置）
bash scripts/evaluate.sh smolvla_pot_multi_v1 100 --task multi

# 2. 学習で見た形
bash scripts/evaluate.sh smolvla_pot_multi_v1 100 --task multi --piece-set carrot_train --name smolvla_pot_multi_v1_seen

# 3. ACT と並べて比較する
bash scripts/analyze_eval.sh act_pot_multi_v1 smolvla_pot_multi_v1 act_pot_multi_v1_seen smolvla_pot_multi_v1_seen --dataset pot_multi_v1
```

- VLA は 1 並列あたりのメモリが大きく（約 3GB）、推論も重いので、評価スクリプトが自動で
  12 並列 × 3 スレッドにする（ACT は 36 並列 × 1 スレッド）。
- 評価は ACT と同じ seed から始めるので、配置（片の数・形・位置）は ACT の評価とまったく同じになる。
- 動画は `outputs/eval/smolvla_pot_multi_v1/videos/`（`view_videos.ipynb` の `FOLDER` を変えて再生）。
