"""Figure 1 of the report: the nested damaged spans of one example 24-word sentence.

The spans follow the damage rule (Sec. 3.2): k words centred on the anchor, k = 1, then 10, 25, 50 and 75% of
the sentence (half-up rounding), each span containing every smaller one. Colours: matplotlib's viridis, a
perceptually uniform sequential colour map, one step per level.

Writes report/overleaf-bln600/figures/nested_spans.pdf. Run: .venv/bin/python scripts/fig_nested_spans.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the "science" matplotlib styles)
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "overleaf-bln600" / "figures" / "nested_spans.pdf"
N_WORDS, ANCHOR = 24, 12  # anchor = 13th word (0-based index 12)
LEVELS = [("1 word", 1, (12, 12)), ("10%", 2, (11, 12)), ("25%", 6, (9, 14)), ("50%", 12, (6, 17)),
          ("75%", 18, (3, 20))]


def main():
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "pdf.fonttype": 42})
    colors = plt.get_cmap("viridis")([0.9, 0.7, 0.5, 0.3, 0.1])
    fig, ax = plt.subplots(figsize=(6.9, 2.55))
    for row, ((name, k, (lo, hi)), color) in enumerate(zip(LEVELS, colors)):
        y = len(LEVELS) - 1 - row
        assert hi - lo + 1 == k and lo <= ANCHOR <= hi
        for i in range(N_WORDS):
            damaged = lo <= i <= hi
            ax.add_patch(Rectangle((i + 0.08, y + 0.14), 0.84, 0.72, facecolor=color if damaged else "white",
                                   edgecolor="#9a9a9a" if not damaged else color, linewidth=0.6))
        ax.add_patch(Rectangle((ANCHOR + 0.08, y + 0.14), 0.84, 0.72, fill=False, edgecolor="black",
                               linewidth=1.3))
        ax.text(-0.5, y + 0.5, name, ha="right", va="center", fontweight="bold")
        ax.text(N_WORDS + 0.3, y + 0.5, f"$k$={k}", ha="left", va="center", fontsize=8.5)
    ax.annotate("anchor (fact token)", xy=(ANCHOR + 0.5, len(LEVELS) - 0.1), xytext=(ANCHOR + 0.5, len(LEVELS) + 0.55),
                ha="center", va="bottom", arrowprops=dict(arrowstyle="-|>", color="black", linewidth=0.8))
    ax.set_xlim(-3.2, N_WORDS + 1.8)
    ax.set_ylim(-0.1, len(LEVELS) + 1.0)
    ax.axis("off")
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(f"written: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
