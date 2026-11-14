#!/usr/bin/env python
"""Crop the numbered figures of the article from its PDF into ``assets/paper_figures``.

The PDF is not part of this repository; pass its path::

    pip install pymupdf
    python scripts/extract_paper_figures.py --pdf /path/to/biomimetics-10-00176.pdf

Each figure is rendered from the page at 200 dpi and cropped to the box listed in
``FIGURES`` (PDF points, origin at the top-left corner of the page).  The boxes enclose
the graphics and sub-panel labels of a figure and exclude its caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# figure number -> (page number as printed, 1-based; crop box x0, y0, x1, y1 in points)
FIGURES: dict[int, tuple[int, tuple[float, float, float, float]]] = {
    1: (7, (94.0, 246.0, 508.5, 546.0)),
    2: (10, (64.0, 110.5, 529.0, 426.0)),
    3: (12, (30.0, 509.0, 554.0, 679.0)),
    4: (14, (38.0, 160.0, 560.0, 704.0)),
    5: (15, (19.0, 73.5, 549.0, 293.5)),
    6: (15, (142.0, 575.5, 488.0, 769.0)),
    7: (16, (39.0, 552.0, 556.0, 722.5)),
    8: (17, (158.0, 483.0, 495.5, 675.0)),
}


def extract(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every figure of ``FIGURES`` to ``<output_dir>/figNN.png``."""
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with fitz.open(pdf_path) as document:
        for number, (page_number, box) in FIGURES.items():
            page = document[page_number - 1]
            pixmap = page.get_pixmap(clip=fitz.Rect(*box), dpi=dpi)
            target = output_dir / f"fig{number:02d}.png"
            pixmap.save(target)
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdf", type=Path, required=True, help="path to the article PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "paper_figures",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    for path in extract(args.pdf, args.output, args.dpi):
        print(f"{path}  ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
