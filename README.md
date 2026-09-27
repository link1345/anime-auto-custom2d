# Anime Auto Custom2D — Nagi Studio

一枚絵から上半身アバターを制作し、ローカルで操作・配信するための技術試作です。
**Issue #1全体は未完了です。** 現在は2.5Dメッシュ・スキニング・輪郭モーフを持つ制作基盤です。カメラ追跡、汎用的な完全自動リギング、生成補完の品質保証は含みません。

## 起動

Python 3.11以上。Windowsでは以下を実行します。

```powershell
./start.ps1
```

または：

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.server
```

[制作画面](http://127.0.0.1:8765)を開くと、同梱のオリジナルキャラNagiを初回に取り込みます。通常の編集・実行はネット接続不要です。`data/`にプロジェクトを保存します。

## 使い方

1. 「一枚絵から」で透過PNGを読み込みます。原画は原寸保存します。
2. パーツを選び「支点・分割」を有効にします。輪郭点のドラッグで分割、クリックで支点を編集します。
3. 「分割・メッシュ候補を生成」→プレビュー→採用します。保護したパーツは上書きしません。分割変更後は採用するまでモデル出力を止めます。
4. 確認モーション、停止、ポーズスライダーで確認します。編集は自動保存され、Undo/Redoとチェックポイントも使用できます。
5. 「差分を追加」で表情PNGを登録します。原画キャンバスと同じ透過差分は表示矩形`0,0,1,1`、部分画像は配置先の正規化矩形を指定します。現在の補間はクロスフェードです。
6. 「モデル出力」で単独で再読込できる`.aac2d` ZIPを保存します。「開く」から再読込します。
7. 配信画面で人／AI／混合と操作元を設定します。人入力はポインターとスライダーです。OBSブラウザソースに`http://127.0.0.1:8765/output`、幅1280、高さ720を指定します。

## See-through / Colab

顔を大きく収めた正方形を登録し、**FFmpegで2160×2160以上**へ拡大できます。Lanczosで画素数を増やす処理です。原画像と処理履歴を別に保存し、既に十分大きい画像は縮小しません。FFmpegはPATH、WindowsのWinGet導入先、または環境変数`AAC2D_FFMPEG`で指定します。

同梱の`assets/nagi-face-source.png`は今回提供された2160×2160のNagi顔素材です。`nagi-face-2160.png`はFFmpegで処理した画像で、元から2160のため画素数は維持しています。

1. 顔素材を登録し「顔素材を重ねて確認」で原画に位置合わせします。
2. 「① Colabノートブック」と「② 顔8種類のジョブ ZIP」または「② 髪・体のジョブ ZIP」を保存します。
3. [Google Colab](https://colab.research.google.com/)でノートブックをアップロードし、GPUをL4/A100へ変更して上から実行します。ローカルのジョブZIPを選びます。
4. 結果ZIPを「③ See-through結果 ZIP」へ読み込み、候補を確認して採用します。入力画像・ジョブID・ハッシュ・寸法を照合し、保護したパーツと手修正を維持します。
5. 重なっている仮パーツは「元の仮パーツを表示」で隠します。未検出の部位は警告されるので、必要な元パーツだけ個別に残せます。

顔の出力は**眉毛・まつげ・口・鼻・肌・耳・瞳・白目の8種類だけ**です。公式モデル内部では全レイヤーを推論し、結果の書き出しで8種類に限定します。2160入力は2176へ余白追加し、結果を2160へ戻します。解像度を落とすフォールバックはありません。

髪・体は前髪、後ろ髪、首、上衣、手袋・手元、下衣、脚衣、靴の公式分類を使います。胸／腕や毛束を必ず自動分離できるわけではありません。「2つに分割」で左右・上下に細分化して部位を設定できます。「独立領域を分割」は透明な縦の隙間で分かれた左右領域が対象です。元レイヤーは非表示・保護して残し、Undoできます。下半身のない原画ではスカートや足を取得できない場合があります。

「顔の要素を追加」から頬の赤み、星・ハートの瞳、任意の透過PNGを重ねられます。表示、強度、サイズ、削除を別レイヤーとして保存します。プリセットの瞳装飾は簡易マスクです。

詳しい設定・復旧方法は[Colab手順](docs/see-through-colab.md)、ノートブック本体は[colab/Nagi_SeeThrough.ipynb](colab/Nagi_SeeThrough.ipynb)です。**Colab上のGPU実推論と生成品質は未検証**です。ローカルのCUDA環境でも実行できます。

```sh
python scripts/run_face_job.py see-through-face-job.zip face-result.zip --repo /path/to/see-through --group-offload
```

## MoGe3・人物推定・キャラクター構築

[構築手順と検証結果](docs/character-construction.md)に一連の操作をまとめています。

- **MoGe3**: 画面からローカルで実行、進捗・ログ確認、中止、結果統合。Colabノートブックと同一ジョブZIPも使用できます。既定は公式`Ruicheng/moge-3-vitl`の固定リビジョン。深度・点群・有効領域・カメラ内部パラメータを保存し、メッシュに深度を反映します。
- **人物・顔・関節**: MediaPipeの画像推論をローカル／Colabで実行。未検出・低信頼・潰れた関節を採用せず、クリックで手動補正できます。
- **リグ**: 関節階層、頂点ごとの最大4ボーンのウェイト、肩・肘・手首の操作。選択した腕を上腕・前腕・手に分割できます。
- **表情**: 透過輪郭の対応または手動対応点から基準メッシュのモーフを生成。WebGLのテクスチャ付き三角形で変形します。
- **目**: See-throughの白目・瞳・まつげを独立描画し、画像のアルファを使って開口・瞳装飾を制限します。
- **髪・隠れ部分**: 頭と肩の円形コライダーによる毛束の角度制約。前面パーツで隠れる境界だけを最大32px延長できます。See-throughの生成結果とも併用します。
- **GLB**: テクスチャ・ボーン・逆バインド行列・ウェイト・モーフターゲットを埋め込んだglTF 2.0を出力します。

Windowsの初回セットアップ（Python 3.12、Git、CUDA対応GPU推奨）:

```powershell
./scripts/setup_compute.ps1 -Python python
```

画面の「ローカル推論の設定」に`.venv-inference/Scripts/python.exe`の絶対パスを設定してください。実行用の依存は制作サーバーの環境から分離しています。

```sh
# ローカルまたはColab/Linuxでも同じワーカーを使用
python scripts/run_depth_job.py moge3-job.zip depth-result.zip
# 手元に重みがある場合
python scripts/run_depth_job.py moge3-job.zip depth-result.zip --checkpoint /path/to/model.pt
python scripts/run_landmark_job.py mediapipe-job.zip landmarks-result.zip
```

2026-09-26、RTX 5060でNagi原画1024×1536のMoGe3推論・画面からの起動・結果統合を確認。MediaPipeは顔478点を検出しましたが、胴体候補は潰れていたため拒否、手は未検出でした。デモの胴体関節は手動補正です。Colab実行自体は未検証です。

## AI操作

`ws://127.0.0.1:8765/ws`に接続してJSONを送ります。`id`に対応する`type: response`と30Hzの`type: frame`が返ります。

```json
{"id":"1","type":"discover"}
{"id":"2","type":"mode","mode":"ai"}
{"id":"3","type":"parameters","source":"ai","values":{"headX":10,"mouthOpen":0.6}}
{"id":"4","type":"motion","name":"wave"}
{"id":"5","type":"expression","expressionId":"登録済みID","strength":0.75,"duration":0.3}
{"id":"6","type":"release"}
{"id":"7","type":"stop"}
```

連続値は1.5秒更新が途絶えると待機に戻ります。表情は`stop`または強度0まで保持します。`release`はAIの連続値を解除します。混合モードの部位ごとの`owners`は`human`/`ai`です。複数AIクライアントが同じ部位を更新する場合は最後の有効な値を使います。接続を認証主体として分離する仕様は未実装です。

## 検証と現状

```sh
python -m unittest discover -s tests -v
```

[実装・検証状況](docs/implementation-status.md)に未実装項目と検証範囲を記載しています。Issueの完了条件と試作の動作確認を混同しないでください。
