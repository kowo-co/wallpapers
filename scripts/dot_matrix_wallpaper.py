#!/usr/bin/env python3
"""
Render a source "plate" image as a code-built LED dot-matrix display.

The plate is composited onto a black canvas at the target wallpaper size,
then sampled onto a coarse grid. Every grid cell becomes one LED: its
radius and glow both scale with that cell's luminance, so bright areas
give big hot bloomed dots and dark areas give small dim (or invisible)
ones. This is NOT a halftone/pixelate filter — dots are drawn as actual
glowing circles with a separate blurred bloom layer.

Run with `uv run --with pillow --with numpy scripts/dot_matrix_wallpaper.py ...`
"""

import argparse
import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


def parse_hex(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def crop_to_content(plate, threshold, pad_frac):
    """Crop the plate to the bounding box of pixels that differ from the
    plate's own corner (background) color by more than `threshold`."""
    arr = np.asarray(plate).astype(np.int16)
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg_arr = corners.mean(axis=0)
    diff = np.abs(arr - bg_arr).max(axis=2)
    mask = diff > threshold
    if not mask.any():
        return plate

    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(plate.width - 1, x1 + pad_x)
    y1 = min(plate.height - 1, y1 + pad_y)

    return plate.crop((x0, y0, x1 + 1, y1 + 1))


def compose_plate(plate, canvas_w, canvas_h, plate_scale, center_y_frac, bg):
    """Fit the plate into the canvas by width, place it at a given vertical
    center fraction, and pad the rest with the background color."""
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)

    target_w = int(canvas_w * plate_scale)
    scale = target_w / plate.width
    target_h = int(plate.height * scale)
    plate_resized = plate.resize((target_w, target_h), Image.LANCZOS)

    x = (canvas_w - target_w) // 2
    cy = int(canvas_h * center_y_frac)
    y = cy - target_h // 2

    canvas.paste(plate_resized, (x, y))
    return canvas


def boost_saturation(img, factor):
    return ImageEnhance.Color(img).enhance(factor)


def sample_grid(canvas, cols, rows):
    """Downsample to cols x rows; each output pixel is that cell's average color."""
    small = canvas.resize((cols, rows), Image.LANCZOS)
    return np.asarray(small).astype(np.float32)


def luminance(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def render_leds(
    cell_rgb,
    canvas_w,
    canvas_h,
    cols,
    rows,
    bg,
    black_point,
    gamma,
    min_radius_frac,
    max_radius_frac,
    glow_strength,
    glow_radius_mult,
    glow_blur_frac,
    hot_boost,
    supersample,
):
    sw, sh = canvas_w * supersample, canvas_h * supersample
    cell_w = sw / cols
    cell_h = sh / rows
    cell = min(cell_w, cell_h)

    lum = luminance(cell_rgb)
    crushed = np.clip((lum - black_point) / max(1.0, 255 - black_point), 0, 1)
    level = crushed ** gamma

    glow_layer = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    core_layer = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    core_draw = ImageDraw.Draw(core_layer)

    min_r = min_radius_frac * cell / 2
    max_r = max_radius_frac * cell / 2

    for j in range(rows):
        for i in range(cols):
            lv = float(level[j, i])
            if lv <= 0.01:
                continue
            r, g, b = [float(c) for c in cell_rgb[j, i, :3]]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            v = min(1.0, v + hot_boost * (lv ** 3))
            r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
            color = (int(r2 * 255), int(g2 * 255), int(b2 * 255))

            radius = min_r + (max_r - min_r) * lv
            cx = (i + 0.5) * cell_w
            cy = (j + 0.5) * cell_h

            glow_r = radius * glow_radius_mult
            glow_alpha = int(255 * glow_strength * lv)
            if glow_alpha > 0 and glow_r > 0:
                glow_draw.ellipse(
                    [cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
                    fill=(*color, glow_alpha),
                )

            core_alpha = int(255 * min(1.0, 0.55 + 0.45 * lv))
            core_draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(*color, core_alpha),
            )

    blur_radius = max(1.0, cell * glow_blur_frac)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(blur_radius))

    out = Image.new("RGBA", (sw, sh), (*bg, 255))
    out.alpha_composite(glow_layer)
    out.alpha_composite(core_layer)
    out = out.convert("RGB")

    out = out.resize((canvas_w, canvas_h), Image.LANCZOS)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="source plate image")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--canvas-width", type=int, required=True)
    p.add_argument("--canvas-height", type=int, required=True)
    p.add_argument("--cols", type=int, default=135, help="grid columns")
    p.add_argument("--rows", type=int, default=0, help="grid rows (0 = auto from canvas aspect)")
    p.add_argument("--plate-scale", type=float, default=0.94, help="plate width as fraction of canvas width")
    p.add_argument("--plate-center-y", type=float, default=0.40, help="plate vertical center as fraction of canvas height")
    p.add_argument("--auto-crop", action="store_true", default=True, help="crop plate to content bounding box before compositing")
    p.add_argument("--no-auto-crop", dest="auto_crop", action="store_false")
    p.add_argument("--crop-threshold", type=float, default=18, help="max-channel diff from bg to count as content")
    p.add_argument("--crop-pad-frac", type=float, default=0.04, help="padding around content bbox as fraction of bbox size")
    p.add_argument("--bg", default="#050505", help="background hex color")
    p.add_argument("--black-point", type=float, default=60, help="luminance (0-255) crushed to zero")
    p.add_argument("--gamma", type=float, default=0.85, help="radius response curve exponent")
    p.add_argument("--min-radius-frac", type=float, default=0.06, help="min LED radius as fraction of cell size")
    p.add_argument("--max-radius-frac", type=float, default=0.96, help="max LED radius as fraction of cell size")
    p.add_argument("--glow-strength", type=float, default=0.85, help="bloom layer opacity multiplier")
    p.add_argument("--glow-radius-mult", type=float, default=2.6, help="bloom circle radius multiplier vs core radius")
    p.add_argument("--glow-blur-frac", type=float, default=0.55, help="gaussian blur sigma as fraction of cell size")
    p.add_argument("--hot-boost", type=float, default=0.35, help="extra brightness pushed into the brightest cells")
    p.add_argument("--saturation", type=float, default=1.35, help="pre-sample saturation multiplier")
    p.add_argument("--supersample", type=int, default=3, help="render scale factor for antialiasing")
    args = p.parse_args()

    bg = parse_hex(args.bg)
    rows = args.rows or round(args.cols * args.canvas_height / args.canvas_width)

    plate = Image.open(args.input).convert("RGB")
    if args.auto_crop:
        plate = crop_to_content(plate, args.crop_threshold, args.crop_pad_frac)
    canvas = compose_plate(plate, args.canvas_width, args.canvas_height, args.plate_scale, args.plate_center_y, bg)
    canvas = boost_saturation(canvas, args.saturation)

    cell_rgb = sample_grid(canvas, args.cols, rows)

    out = render_leds(
        cell_rgb,
        args.canvas_width,
        args.canvas_height,
        args.cols,
        rows,
        bg,
        args.black_point,
        args.gamma,
        args.min_radius_frac,
        args.max_radius_frac,
        args.glow_strength,
        args.glow_radius_mult,
        args.glow_blur_frac,
        args.hot_boost,
        args.supersample,
    )
    out.save(args.output)
    print(f"{args.output}: {args.canvas_width}x{args.canvas_height} grid {args.cols}x{rows}")


if __name__ == "__main__":
    main()
