#!/usr/bin/env python3
"""
make_icon.py — generate LocalFlow's app icon (soundwave bars on a calm squircle).

Run with SYSTEM python3 (needs Pillow — a build-only dep we keep OUT of
requirements.txt so the runtime venv stays lean):

    python3 make_icon.py

Outputs, next to this file:
    icon_preview.png   1024px master, for eyeballing
    LocalFlow.icns     the multi-resolution bundle icon (make_app.sh copies it in)

Design law (learning/design-identity.md): calm surface, ONE accent. The bars are
the single accent; the ground stays mono. Change ACCENT below to re-colour — that
is the only knob a red-pen should ever need.
"""
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

# ---- the only knobs -------------------------------------------------------
ACCENT   = (84, 199, 184)     # calm teal — the ONE accent (the waveform)
GROUND_1 = (44, 44, 48)       # squircle top  (subtle vertical gradient…)
GROUND_2 = (24, 24, 27)       # squircle base (…so it reads calm, not flat/dead)
# symmetric equaliser heights, 0..1 — reads as "voice" at a glance
BARS = [0.26, 0.46, 0.72, 0.92, 1.00, 0.92, 0.72, 0.46, 0.26]

S = 1024                      # master canvas
INSET = 100                   # macOS Big Sur icon grid: 824x824 art in 1024
RADIUS = 185                  # …with this corner radius


def rounded_mask(size, box, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle(box, radius=radius, fill=255)
    return m


def vertical_gradient(size, top, bottom):
    grad = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        grad.putpixel((0, y), tuple(
            round(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return grad.resize((size, size))


def draw_master():
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # the calm ground: gradient clipped to the squircle
    box = (INSET, INSET, S - INSET, S - INSET)
    ground = vertical_gradient(S, GROUND_1, GROUND_2).convert("RGBA")
    canvas.paste(ground, (0, 0), rounded_mask(S, box, RADIUS))

    # the waveform: rounded bars, symmetric about the vertical centre
    art_w = (S - 2 * INSET)
    n = len(BARS)
    gap_ratio = 0.55                     # bar : gap — airy, not crammed
    unit = art_w * 0.62 / (n + (n - 1) * gap_ratio)
    bar_w = unit
    gap = unit * gap_ratio
    total = n * bar_w + (n - 1) * gap
    x = (S - total) / 2
    cy = S / 2
    max_h = (S - 2 * INSET) * 0.52
    d = ImageDraw.Draw(canvas)
    for h in BARS:
        half = max_h * h / 2
        d.rounded_rectangle((x, cy - half, x + bar_w, cy + half),
                            radius=bar_w / 2, fill=ACCENT + (255,))
        x += bar_w + gap
    return canvas


def build_icns(master):
    iconset = HERE / "LocalFlow.iconset"
    iconset.mkdir(exist_ok=True)
    specs = [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1),
             (128, 2), (256, 1), (256, 2), (512, 1), (512, 2)]
    for base, scale in specs:
        px = base * scale
        name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
        master.resize((px, px), Image.LANCZOS).save(iconset / name)
    subprocess.run(["iconutil", "-c", "icns", str(iconset),
                    "-o", str(HERE / "LocalFlow.icns")], check=True)
    for p in iconset.iterdir():
        p.unlink()
    iconset.rmdir()


def main():
    master = draw_master()
    master.save(HERE / "icon_preview.png")
    build_icns(master)
    print(f"wrote {HERE/'icon_preview.png'} and {HERE/'LocalFlow.icns'}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError:
        sys.exit("needs Pillow: run with system python3 (pip3 install pillow)")
