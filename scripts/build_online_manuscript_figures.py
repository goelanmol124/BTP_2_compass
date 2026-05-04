#!/usr/bin/env python3
"""Build figure PNGs referenced by COMPASS_JCTC_MANUSCRIPT_ONLINE_TARGET.tex.

Uses existing replay plots under figures/ when dedicated online assets are absent.
Replace outputs with publication figures when available.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def _resize_width(im: Image.Image, width: int) -> Image.Image:
    if im.width == width:
        return im
    r = width / im.width
    h = max(1, int(im.height * r))
    return im.resize((width, h), Image.Resampling.LANCZOS)


def _hstack(a: Image.Image, b: Image.Image, gap: int = 12) -> Image.Image:
    h = max(a.height, b.height)
    w = a.width + gap + b.width
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(a, (0, (h - a.height) // 2))
    out.paste(b, (a.width + gap, (h - b.height) // 2))
    return out


def _vstack(a: Image.Image, b: Image.Image, gap: int = 12) -> Image.Image:
    w = max(a.width, b.width)
    h = a.height + gap + b.height
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(a, ((w - a.width) // 2, 0))
    out.paste(b, ((w - b.width) // 2, a.height + gap))
    return out


def montage_four(paths: list[Path], out: Path, tile_w: int = 1100) -> None:
    tiles = [_resize_width(Image.open(p).convert("RGB"), tile_w) for p in paths]
    top = _hstack(tiles[0], tiles[1])
    bot = _hstack(tiles[2], tiles[3])
    canvas = _vstack(top, bot, gap=16)
    canvas.save(out, dpi=(200, 200))


def main() -> None:
    tica80 = FIG / "replay_tica2d_chunk20_80ns.png"
    spib80 = FIG / "replay_spib1d_80ns.png"
    tica200 = FIG / "replay_tica2d_200ns.png"
    for p in (tica80, spib80, tica200):
        if not p.is_file():
            raise SystemExit(f"Missing source figure: {p}")

    FIG.mkdir(parents=True, exist_ok=True)

    montage_four([tica80, spib80, tica200, tica80], FIG / "online_screening_summary.png")
    _resize_width(Image.open(tica80).convert("RGB"), 2200).save(
        FIG / "online_helix_convergence.png", dpi=(200, 200)
    )
    _resize_width(Image.open(tica200).convert("RGB"), 2200).save(
        FIG / "online_distributional.png", dpi=(200, 200)
    )
    _resize_width(Image.open(spib80).convert("RGB"), 2200).save(
        FIG / "online_trust_controller.png", dpi=(200, 200)
    )
    print("Wrote online_*.png under", FIG)


if __name__ == "__main__":
    main()
