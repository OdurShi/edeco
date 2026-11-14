# Figures of the article

Figures cropped from the published article (without their captions) by
`scripts/extract_paper_figures.py`. Captions are quoted as printed.

| File | Figure | Caption |
|---|---|---|
| `fig01.png` | Figure 1 | Sketch for hybrid search framework. |
| `fig02.png` | Figure 2 | A flowchart of the proposed EDECO. |
| `fig03.png` | Figure 3 | Multiple comparisons using CDV to evaluate ECO, EDECO and its two variants. |
| `fig04.png` | Figure 4 | Rankings based on "Ave" of EDECO and its comparison algorithms on each benchmark function. |
| `fig05.png` | Figure 5 | The visualization of Wilcoxon rank sum test results between EDECO and its competitors (a = 0.05). |
| `fig06.png` | Figure 6 | The visualization of Friedman test results between EDECO and its competitors. |
| `fig07.png` | Figure 7 | Multiple comparisons using CDV to evaluate EDECO and its competitors. |
| `fig08.png` | Figure 8 | The Friedman ranking of EDECO and its competitors on constrained engineering optimization problems. |

Credit: W. Tang, S. Shi, Z. Lu, M. Lin, H. Cheng, "EDECO: An Enhanced Educational
Competition Optimizer for Numerical Optimization Problems", *Biomimetics* 2025, 10(3),
176, <https://doi.org/10.3390/biomimetics10030176>. © 2025 by the authors, licensee
MDPI, Basel, Switzerland; open access under the Creative Commons Attribution
(CC BY 4.0) licence, <https://creativecommons.org/licenses/by/4.0/>.

To regenerate the images from a copy of the article PDF:

```bash
pip install pymupdf
python scripts/extract_paper_figures.py --pdf /path/to/biomimetics-10-00176.pdf
```
