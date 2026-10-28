#!/usr/bin/env python3
"""img2text.py - image -> most faithful text representation.

Decode : ffmpeg (any format/animated first frame/16-bit/HDR-ish), PIL fallback
Resize : gamma-correct (linear light) Lanczos, aspect-corrected for terminal cells
Modes  : half     1x2 px/cell  '▀' fg/bg truecolor (seamless, no gaps between rows)
         quad     2x2 px/cell  best of 14 quadrant glyphs, optimal 2-colour fit
         braille  2x4 px/cell  8 coloured dots + bg (highest spatial resolution)
         ascii    1x2 px/cell  luminance ramp, coloured (pure ASCII glyphs)
Output : ansi (truecolor) | html | txt (no colour)

Usage: img2text.py photo.webp -m quad -c 160 > out.ans ; cat out.ans
       img2text.py photo.jpg -m braille -f html -o out.html
"""
import argparse, html, io, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

CELL = {"half": (1, 2), "quad": (2, 2), "braille": (2, 4), "ascii": (1, 2)}
RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(path, bg):
    img = None
    if shutil.which("ffmpeg"):
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
        if p.returncode == 0 and p.stdout:
            img = Image.open(io.BytesIO(p.stdout))
    if img is None:
        img = ImageOps.exif_transpose(Image.open(path))
    img.load()
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        base = Image.new("RGBA", img.size, (*bg, 255))
        base.alpha_composite(img)
        img = base
    return img.convert("RGB")


def resize_linear(img, w, h):
    """Downscale in linear light, return sRGB float32 (h,w,3) 0..255."""
    a = np.asarray(img, np.float32) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)
    out = np.stack([np.asarray(Image.fromarray(lin[..., c], mode="F").resize((w, h), Image.LANCZOS))
                    for c in range(3)], -1)
    out = np.clip(out, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    return np.clip(srgb * 255, 0, 255).astype(np.float32)


def prepare(img, cols, mode, contrast, saturation, sharpen):
    cw, ch = CELL[mode]
    W, H = img.size
    pw = cols * cw
    rows = max(1, round(H * pw / W / ch))        # cell ratio 1:2 => px stay square
    ph = rows * ch
    px = resize_linear(img, pw, ph)
    im = Image.fromarray(px.astype(np.uint8))
    if saturation != 1: im = ImageEnhance.Color(im).enhance(saturation)
    if contrast != 1:   im = ImageEnhance.Contrast(im).enhance(contrast)
    if sharpen:         im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=int(sharpen * 100), threshold=2))
    return np.asarray(im, np.float32), rows, cols


def cells(px, rows, cols, cw, ch):
    """(rows*ch, cols*cw, 3) -> (rows, cols, ch*cw, 3), row-major inside cell."""
    return (px.reshape(rows, ch, cols, cw, 3).transpose(0, 2, 1, 3, 4)
              .reshape(rows, cols, ch * cw, 3))


def u8(a): return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def m_half(px, R, C):
    c = cells(px, R, C, 1, 2)
    return np.full((R, C), "▀", object), u8(c[:, :, 0]), u8(c[:, :, 1])


def m_quad(px, R, C):
    c = cells(px, R, C, 2, 2)                              # px order UL,UR,LL,LR
    glyphs = np.array(list(" ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"), object)
    masks = np.array([[(k >> i) & 1 for i in range(4)] for k in range(1, 15)], np.float32)
    n = masks.sum(1)
    tot = c.sum(2)
    fs = np.einsum("kp,rcpz->rckz", masks, c)
    bs = tot[:, :, None, :] - fs
    fg = fs / n[None, None, :, None]
    bg = bs / (4 - n)[None, None, :, None]
    score = n * (fg ** 2).sum(-1) + (4 - n) * (bg ** 2).sum(-1)   # maximise == minimise SSE
    k = score.argmax(2)
    r, q = np.indices(k.shape)
    return glyphs[k + 1], u8(fg[r, q, k]), u8(bg[r, q, k])


BRAILLE_W = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], np.int32).reshape(8)


def m_braille(px, R, C):
    c = cells(px, R, C, 2, 4)                              # (R,C,8,3)
    lum = c @ LUMA
    thr = lum.mean(2, keepdims=True)
    for _ in range(2):                                     # 2-means refinement
        on = lum > thr
        mo = np.where(on, lum, 0).sum(2) / np.maximum(on.sum(2), 1)
        mf = np.where(~on, lum, 0).sum(2) / np.maximum((~on).sum(2), 1)
        thr = ((mo + mf) / 2)[..., None]
    on = lum > thr
    non, noff = on.sum(2, keepdims=True), (~on).sum(2, keepdims=True)
    mean = c.mean(2)
    fg = np.where(non > 0, (c * on[..., None]).sum(2) / np.maximum(non, 1), mean[:, :, None, :].squeeze(2))
    bg = np.where(noff > 0, (c * (~on)[..., None]).sum(2) / np.maximum(noff, 1), mean)
    bits = (on * BRAILLE_W).sum(2)
    chars = np.vectorize(lambda b: chr(0x2800 + int(b)), otypes=[object])(bits)
    return chars, u8(fg), u8(bg)


def m_ascii(px, R, C):
    c = cells(px, R, C, 1, 2).mean(2)
    lum = (c @ LUMA) / 255
    idx = np.clip((lum ** 0.9 * (len(RAMP) - 1)).round().astype(int), 0, len(RAMP) - 1)
    chars = np.array(list(RAMP), object)[idx]
    fg = c * (255 / np.maximum(c.max(-1, keepdims=True), 1))   # keep hue, full brightness
    return chars, u8(fg), np.zeros((R, C, 3), np.uint8)


MODES = {"half": m_half, "quad": m_quad, "braille": m_braille, "ascii": m_ascii}


def to_ansi(ch, fg, bg):
    out = []
    for r in range(ch.shape[0]):
        prev, line = None, []
        for c in range(ch.shape[1]):
            key = (tuple(fg[r, c]), tuple(bg[r, c]))
            if key != prev:
                f, b = key
                line.append(f"\x1b[38;2;{f[0]};{f[1]};{f[2]};48;2;{b[0]};{b[1]};{b[2]}m")
                prev = key
            line.append(ch[r, c])
        out.append("".join(line) + "\x1b[0m")
    return "\n".join(out) + "\n"


def to_html(ch, fg, bg, title):
    rows = []
    for r in range(ch.shape[0]):
        prev, buf, line = None, [], []
        def flush():
            if buf:
                f, b = prev
                line.append(f'<span style="color:#{f[0]:02x}{f[1]:02x}{f[2]:02x};'
                            f'background:#{b[0]:02x}{b[1]:02x}{b[2]:02x}">{html.escape("".join(buf))}</span>')
        for c in range(ch.shape[1]):
            key = (tuple(fg[r, c]), tuple(bg[r, c]))
            if key != prev:
                flush(); buf = []; prev = key
            buf.append(ch[r, c])
        flush()
        rows.append("".join(line))
    return ("<!doctype html><meta charset=utf-8><title>%s</title><style>body{margin:0;background:#000}"
            "pre{margin:0;font:12px/1 'DejaVu Sans Mono','Menlo','Consolas',monospace;letter-spacing:0;"
            "white-space:pre}</style><pre>%s</pre>" % (html.escape(title), "\n".join(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-m", "--mode", choices=MODES, default="quad")
    ap.add_argument("-c", "--cols", type=int, default=0, help="width in characters (default: terminal width)")
    ap.add_argument("-f", "--format", choices=["ansi", "html", "txt"], default="ansi")
    ap.add_argument("-o", "--out", help="output file (default stdout)")
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--saturation", type=float, default=1.0)
    ap.add_argument("--sharpen", type=float, default=0.3, help="unsharp amount, 0 = off")
    ap.add_argument("--bg", default="0,0,0", help="background for transparent images, r,g,b")
    a = ap.parse_args()

    cols = a.cols or shutil.get_terminal_size((120, 40)).columns
    img = load(a.image, tuple(int(v) for v in a.bg.split(",")))
    px, R, C = prepare(img, cols, a.mode, a.contrast, a.saturation, a.sharpen)
    ch, fg, bg = MODES[a.mode](px, R, C)

    if a.format == "ansi":   text = to_ansi(ch, fg, bg)
    elif a.format == "html": text = to_html(ch, fg, bg, a.image)
    else:                    text = "\n".join("".join(r) for r in ch) + "\n"

    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
        print(f"{C}x{R} chars -> {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
