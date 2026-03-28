from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from toroidal_rotate import toroidal_rotate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "showcase.gif"


def make_grayscale_example(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    image = ((xx * 5 + yy * 3) % 256).astype(np.uint8)
    image[((xx // 12) + (yy // 12)) % 2 == 0] = np.clip(
        image[((xx // 12) + (yy // 12)) % 2 == 0] + 40, 0, 255
    )
    return image


def make_rgb_example(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = (xx * 255 // max(1, size - 1)).astype(np.uint8)
    image[..., 1] = (yy * 255 // max(1, size - 1)).astype(np.uint8)
    image[..., 2] = (((xx // 12 + yy // 12) % 2) * 180).astype(np.uint8)
    return image


def to_rgb(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image, mode="L").convert("RGB")
    return Image.fromarray(image, mode="RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rounded_rectangle((6, 6, 90, 24), radius=6, fill=(0, 0, 0))
    draw.text((12, 10), label, fill=(255, 255, 255))
    return out


def make_frame(angle: float) -> Image.Image:
    gray = make_grayscale_example()
    rgb = make_rgb_example()

    gray_rot = toroidal_rotate(gray, angle)
    rgb_rot = toroidal_rotate(rgb, angle)

    left_top = add_label(to_rgb(gray), "gray input")
    right_top = add_label(to_rgb(gray_rot), f"gray {int(angle):+d} deg")
    left_bottom = add_label(to_rgb(rgb), "rgb input")
    right_bottom = add_label(to_rgb(rgb_rot), f"rgb {int(angle):+d} deg")

    gap = 8
    panel_w, panel_h = left_top.size
    canvas = Image.new(
        "RGB",
        (panel_w * 2 + gap * 3, panel_h * 2 + gap * 3 + 28),
        (245, 245, 240),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), "Toroidal pseudo-rotation from the README examples", fill=(20, 20, 20))

    y0 = 28 + gap
    canvas.paste(left_top, (gap, y0))
    canvas.paste(right_top, (gap * 2 + panel_w, y0))
    canvas.paste(left_bottom, (gap, y0 + gap + panel_h))
    canvas.paste(right_bottom, (gap * 2 + panel_w, y0 + gap + panel_h))
    return canvas


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    angles = list(range(-30, 31, 6)) + list(range(24, -31, -6))
    frames = [make_frame(angle) for angle in angles]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
