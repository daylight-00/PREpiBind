#!/usr/bin/env python3
"""Render the 7NZF molecular assets used in the architecture figure.

The three outputs intentionally share the exact same experimental structure,
camera, crop, and molecular representations. Only the palette changes so the
architecture figure can visually distinguish the input being emphasized.

Outputs
-------
7NZF_HLA_focus.png
    HLA alpha/beta chains emphasized; peptide de-emphasized with neutral gray.
7NZF_epitope_focus.png
    Peptide emphasized; HLA alpha/beta chains de-emphasized with neutral gray.
7NZF_structure_prediction_panel.png
    Fully colored complex for the structure-prediction-derived representation
    panel, using the teal/mint palette of that panel.

Example
-------
python render_7nzf_architecture_assets.py /path/to/7NZF.cif \
    --output-dir ../../figures/architecture_assets

Dependencies
------------
- pymol-open-source (Python package providing pymol2)
- Pillow

The 7NZF mmCIF used while preparing the manuscript has auth chain IDs
AAA (HLA alpha), BBB (HLA beta), and CCC (peptide).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import pymol2


@dataclass(frozen=True)
class Palette:
    hla_alpha: str
    hla_beta: str
    peptide: str


PALETTES = {
    "hla_focus": Palette(
        hla_alpha="0xA9BEDA",
        hla_beta="0x3F689D",
        peptide="0xD6D8DC",
    ),
    "epitope_focus": Palette(
        hla_alpha="0xCDD2D9",
        hla_beta="0xAEB7C2",
        peptide="0xE6952E",
    ),
    "structure_prediction": Palette(
        hla_alpha="0xB7D3CF",
        hla_beta="0x6E9AA1",
        peptide="0xD7A55A",
    ),
}

OUTPUT_NAMES = {
    "hla_focus": "7NZF_HLA_focus.png",
    "epitope_focus": "7NZF_epitope_focus.png",
    "structure_prediction": "7NZF_structure_prediction_panel.png",
}


# Keep these selections in one place. 7NZF.cif from RCSB uses the auth chain
# IDs below when loaded by PyMOL.
HLA_ALPHA = "chain AAA"
HLA_BETA = "chain BBB"
PEPTIDE = "chain CCC and not resn HOH"
COMPLEX = "chain AAA+BBB+CCC and not resn HOH"


def _configure_scene(cmd, cif_path: Path) -> None:
    """Load 7NZF and establish the master representation/camera."""
    cmd.reinitialize()
    cmd.load(str(cif_path), "cpx")
    cmd.remove("solvent")
    cmd.hide("everything")

    cmd.show("cartoon", f"({HLA_ALPHA}) or ({HLA_BETA})")
    cmd.show("sticks", PEPTIDE)
    cmd.set("stick_radius", 0.22, PEPTIDE)

    # Smooth, publication-oriented molecular cartoon settings.
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_sampling", 14)

    # Transparent background and restrained lighting for Figma placement.
    cmd.set("ray_opaque_background", 0)
    cmd.bg_color("white")
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("specular", 0.08)
    cmd.set("shininess", 15)
    cmd.set("ambient", 0.62)
    cmd.set("direct", 0.38)
    cmd.set("antialias", 2)
    cmd.set("depth_cue", 0)

    # Master camera. Every asset is rendered without moving the camera again.
    cmd.orient(COMPLEX)
    cmd.zoom(COMPLEX, 3)
    cmd.turn("x", 30)
    cmd.turn("y", 45)


def _apply_palette(cmd, palette: Palette) -> None:
    cmd.color(palette.hla_alpha, HLA_ALPHA)
    cmd.color(palette.hla_beta, HLA_BETA)
    cmd.color(palette.peptide, PEPTIDE)


def _crop_transparent_square(path: Path, padding: int = 70) -> None:
    """Crop alpha bounds and recenter on a square transparent canvas."""
    image = Image.open(path).convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(image.width, x1 + padding)
    y1 = min(image.height, y1 + padding)

    crop = image.crop((x0, y0, x1, y1))
    side = max(crop.width, crop.height)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    x = (side - crop.width) // 2
    y = (side - crop.height) // 2
    canvas.alpha_composite(crop, (x, y))
    canvas.save(path)


def render_all(
    cif_path: Path,
    output_dir: Path,
    size: int = 1600,
    dpi: int = 300,
    padding: int = 70,
) -> list[Path]:
    """Render all three architecture assets from one shared PyMOL scene."""
    cif_path = cif_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    with pymol2.PyMOL() as pymol:
        cmd = pymol.cmd
        _configure_scene(cmd, cif_path)

        for key in ("hla_focus", "epitope_focus", "structure_prediction"):
            _apply_palette(cmd, PALETTES[key])
            output = output_dir / OUTPUT_NAMES[key]
            cmd.png(
                str(output),
                width=size,
                height=size,
                dpi=dpi,
                ray=1,
            )
            rendered.append(output)

    for output in rendered:
        _crop_transparent_square(output, padding=padding)

    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the three 7NZF assets used in the model architecture figure."
    )
    parser.add_argument("cif", type=Path, help="Path to the coordinate file 7NZF.cif")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current directory)",
    )
    parser.add_argument("--size", type=int, default=1600, help="Square ray-render size")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI metadata")
    parser.add_argument(
        "--padding",
        type=int,
        default=70,
        help="Transparent pixel padding after alpha crop",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_all(
        args.cif,
        args.output_dir,
        size=args.size,
        dpi=args.dpi,
        padding=args.padding,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
