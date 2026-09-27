# ナギの表情登録用素材

生成方式: built-in image_gen。基準: nagi-neutral-reference.png。

- nagi-blink-2160.png: 閉じ目。抽出対象はまつげ（必要なら肌も）。
- nagi-mouth-open-2160.png: あ口。抽出対象は口。
- *-source.png: 生成原寸1254×1254。

登録用2枚はFFmpeg Lanczosで2160×2160へ拡大したRGBA PNG。透明背景を保持。細部の復元ではない。既に2160pxなので登録時の拡大はOFFでよい。
元の正面顔は別保存。生成に伴う輪郭・配置の微差があるため、抽出後のフィットでは目頭・目尻／口角の対応点を確認する。See-through実推論・モーフの品質確認は未実施。

## 最終生成プロンプト

### 閉じ目
Edit this exact supplied Nagi character face image for an animation blink target. ONE square image only, no sheet, no text. Preserve original face framing and size, front-facing angle, hair silhouette and strands, mint hair, face outline, ears, skin shading, eyebrows, nose, neck and small CLOSED smiling mouth exactly. Change ONLY both eyes to relaxed fully closed blink: upper eyelids descend to meet lower lids, thin gently curved dark brown lash bands with visible finite thickness and original outer lash tips, natural relaxed closure rather than upside down happy crescents. No iris or white visible. Keep original eye inner/outer corner positions and widths. Transparent background, preserve alpha. Closely match source so used as morph reference and See-through extraction input. Maximum available square resolution.

### 開き口
Precise expression edit of supplied Nagi anime face for mouth animation reference. ONE square face closeup, same original framing and front-facing orientation. Preserve mint hair, golden eyes OPEN, eyelids, eyebrows, nose, chin outline, ears, neck, colors and all placements. Change ONLY the small closed smile to a moderately open speaking 'ah' mouth. Preserve mouth center and width, move lower lip downward slightly, simple clean rounded mouth opening, dark warm interior and small softly pink tongue at bottom, subtle narrow upper teeth optional. Not huge laugh, not surprised O, no exaggerated blush, no added face motion. Keep the rest of face as close as possible to the original. Transparent background preserving alpha. This is a reference image for extracting the mouth and fitting the original closed mouth mesh into the open shape. Highest available square resolution, no text, no sheet.
