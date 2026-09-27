# デモ素材

本プロジェクト用に組み込みimage_genで生成したオリジナルキャラクター「Nagi / ナギ」。外部キャラクター素材のコピーではない。最初のピンク髪の参考画像は同梱していない。今回提供されたNagiの顔画像は以下の通り同梱する。

- `nagi.png`: 1024×1536 RGBA。上半身・腕のデモ原画。入力透明度は保持。
- `nagi-face-draft.png`: 1254×1254 RGBA。顔アップの下書き。要求した3072×3072で出力されなかったため、未処理では2160以上の入力条件を満たさない。現在は利用者指定によりFFmpeg拡大を選択できる。

## 生成プロンプト（組み込みツール、CLI/APIは不使用）

### nagi.png

Create an original anime upper-body avatar reference illustration for a desktop puppet editor, transparent background. One adult young woman, front-facing symmetrical neutral pose, from top of head through waist, both arms slightly away from torso and both hands completely visible, no cropped hair or hands. Soft mint teal medium-length hair with two clearly separated side locks, neat bangs, golden amber eyes, gentle small closed smile, ivory blouse with navy sailor collar and small coral ribbon. Clean dark outlines, crisp flat cel shading, simple readable separate silhouettes, polished charming Japanese character illustration. Centered with generous clear margin. No text, no props, no watermark. This will be the original demo character named Nagi.

### nagi-face-draft.png

Create a high-resolution FACE-ONLY reference of this same original character Nagi for See-through facial part segmentation. Required output canvas is SQUARE 3072 x 3072 pixels (at minimum 2160 x 2160), and the face itself must fill most of the square. Crop composition from hair above the forehead to below the chin; both eyes, eyebrows, nose, mouth, cheeks and chin are fully visible. Keep identical mint teal hair, amber eyes, original face design, neutral open eyes and small closed smile. Crisp clean detailed lineart and flat cel shading; no glow or background haze. True transparent background. This is a dedicated high-information facial asset, not the full-body image scaled up. Preserve identity and front-facing angle exactly.

- `nagi-face-source.png`: 今回利用者が提供した `export.png` のコピー。2160×2160 RGBA。
- `nagi-face-2160.png`: 上記をFFmpegのLanczos / RGBA指定で処理。元から2160のため拡大不要。原画像と同じ画素を保持。
