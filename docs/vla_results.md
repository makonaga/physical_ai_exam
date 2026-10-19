# VLA（SmolVLA）の結果：散らばった食材片から一番近い片を鍋に入れる

手順と SmolVLA を選んだ理由は [vla_howto.md](vla_howto.md)。学習データと評価の条件は段階2（ACT）と同じ
（[step2_results.md](step2_results.md)）。SageMaker Studio で実行した結果。

## 条件

| 項目 | 内容 |
|---|---|
| モデル | SmolVLA（`lerobot/smolvla_base` から追加学習）。全体 4.5 億パラメータのうち、行動を出す部分（action expert）の 1 億だけを学習 |
| 学習データ | `pot_multi_v1`（成功 1,000 エピソード、170,072 フレーム。ACT と同じ） |
| 入力 | 俯瞰カメラ・手首カメラ画像（224×224 を 512×512 に拡大）、6 関節の角度、指示文 "Put the food piece closest to the robot into the pot." |
| 出力 | 6 関節の目標角度 50 ステップ分（2 秒）。flow matching で 10 回かけて出す |
| カメラの対応 | overhead → `observation.images.camera1`、wrist → `observation.images.camera2`（事前学習済みモデルのカメラ名） |

## 1. 学習（ml.g5.2xlarge、NVIDIA A10G × 1）

| 項目 | 値 |
|---|---|
| ステップ数 | 20,000（バッチ 64、約 7.5 エポック。サンプル数の合計 128 万は ACT と同じ） |
| 所要時間 | 6 時間 54 分（1.24 秒/ステップ、52 サンプル/秒）+ 事前学習済みモデルの読み込み 3.5 分 |
| GPU メモリ | 14.0GB（24GB 中） |
| loss | 0.737（100 ステップ）→ 0.052（5,000）→ 0.035（10,000）→ 0.027（15,000）→ 0.025（20,000） |
| 学習率 | 最大 1e-4（666 ステップで上がり、20,000 ステップで 2.5e-6 まで下がる） |

試運転（ml.g5.xlarge、バッチ 32、200 ステップ）は 0.63 秒/ステップ、GPU メモリ 7.7GB だった。

loss は ACT と定義が違う（ACT は行動の L1 誤差、SmolVLA は flow matching の誤差）ので、値を直接は比べられない。

## 2. 評価

（実施待ち。ACT と同じ 100 の配置で、初めて見る形・学習で見た形の 2 種類を評価する）
