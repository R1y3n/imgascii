# ImageASCII convert various image formats to colored ASCII text
---
**Pipeline**
1. **Decode:** ffmpeg reads any format, including webp, avif, and the first frame of an animated gif. It falls back to Pillow, and transparent images are composited onto a background colour.
2. **Resize:** Lanczos downscale in linear light (gamma-correct), with the cell aspect ratio taken into account. Optional sharpen, contrast, and saturation are applied afterwards.
3. **Cell fitting:** each character cell is fitted to its pixels by one of four modes (below).
4. **Output:** truecolor ANSI, HTML, or plain text.

**Modes (`-m`)**
- `half`: one `▀` per cell with a foreground and background colour, so each cell shows two pixels. Rows join up with no gaps.
- `quad` (default): 2×2 pixels per cell. It picks the best of 14 quadrant glyphs with the optimal two-colour fit, so edges are sharper than `half`.
- `braille`: 2×4 pixels per cell, the highest spatial resolution. Dots are split with an iterated two-means threshold and coloured separately from the background.
- `ascii`: a coloured luminance ramp using pure ASCII glyphs.

**Usage**
```
python3 img2text.py photo.webp -m quad -c 160 > out.ans && cat out.ans
python3 img2text.py photo.jpg -m braille -f html -o out.html
```
Use a truecolor terminal and a font that has the block and braille glyphs, such as DejaVu Sans Mono. The HTML output uses `line-height:1` so rows stay seamless.

**Limits**
- `-f txt` has no colour, so it is only a rough approximation, especially in braille mode.
