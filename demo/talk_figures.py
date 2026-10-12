"""The two figures of the demo's talk mode, drawn with matplotlib in the style of the paper's Figure 2
(SciencePlots "science" style, Times, Okabe-Ito colours and one marker per method), returned as SVG text.
Sizes and fonts are larger than in the paper because the figures are shown on a projector."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import scienceplots  # noqa: F401  (registers the "science" styles)
    STYLE = ["science", "no-latex"]
except ImportError:  # the figures still build, with matplotlib's default style
    STYLE = ["default"]

INK, MUTED = "#0b0b0b", "#52514e"
LEVELS = ["1w", "10", "25", "50", "75"]
TICKS = ["1 word", "10%", "25%", "50%", "75%"]
# as in scripts/paper_results.py
MAIN = [("bertdict", "BERT + dict.", "#0072B2", "o"), ("llm", "LLM few-shot", "#D55E00", "s"),
        ("lookup", "Lookup", "#E69F00", "^"), ("bart", "BART", "#009E73", "D")]


def _style():
    plt.style.use(STYLE)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "font.size": 15, "axes.linewidth": 0.9, "svg.fonttype": "none"})


def _svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg"):]


def _p(p):
    return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")


def levels_figure(ci, tests):
    """Fact Recovery Rate and anchor recovery by level with 95% intervals; the Holm-adjusted McNemar p-value
    of BERT + dict. vs. LLM (H2) is written above the anchor panel where it is below 0.05."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(LEVELS))
    for ax, (metric, title) in zip(axes, [("frr", "Fact Recovery Rate"), ("anchor", "Anchor recovery")]):
        ax.plot(x, [ci[metric]["none"][lv][0] for lv in LEVELS], color=MUTED, linewidth=1.3, linestyle="--",
                label="no repair")
        for key, name, color, mark in MAIN:
            v, lo, hi = (np.array([ci[metric][key][lv][i] for lv in LEVELS]) for i in range(3))
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, v, color=color, linewidth=2.2, marker=mark, markersize=7, label=name)
        ax.set_xticks(x, TICKS)
        ax.minorticks_off()
        ax.tick_params(top=False, right=False)
        ax.set_ylim(0, 0.8)
        ax.set_xlim(-0.3, 4.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ["0%", "20%", "40%", "60%", "80%"])
        ax.set_title(title, loc="left", color=INK, fontsize=16)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    anchor = axes[1]
    for i, lv in enumerate(LEVELS):
        t = next(t for t in tests if t["pair"] == ["bertdict", "llm"] and t["level"] == lv)
        if t["mcnemar_holm"] < 0.05:
            anchor.text(i, 0.72, _p(t["mcnemar_holm"]), ha="center", fontsize=13, style="italic", color=INK)
    anchor.text(4.25, 0.785, "BERT + dict. vs. LLM, Holm-adjusted McNemar", ha="right", fontsize=11.5, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.07), fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.5)
    return _svg(fig)


def strip_figure(sentences, method, level, highlight_id, method_name):
    """Span BERTScore of every repair of one method at one level, split by whether the anchor was restored.
    The repair of the talk's example sentence is circled."""
    _style()
    rows = [(s["repairs"][level][method]["bs"], s["repairs"][level][method]["anchor"], s["id"]) for s in sentences]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    rng = np.random.default_rng(0)
    for ok, y, color, label in [(1, 1, "#0072B2", "anchor restored"), (0, 0, "#D55E00", "anchor wrong")]:
        pts = [(bs, sid) for bs, a, sid in rows if a == ok]
        xs = np.array([bs for bs, _ in pts])
        ys = y + rng.uniform(-0.22, 0.22, len(xs))
        ax.scatter(xs, ys, s=46, facecolors=color if ok else "none", edgecolors=color, linewidths=1.4,
                   label=f"{label} ({len(xs)})", zorder=3)
        for (bs, sid), yy in zip(pts, ys):
            if sid == highlight_id:
                ax.scatter([bs], [yy], s=260, facecolors="none", edgecolors=INK, linewidths=1.8, zorder=4)
    ax.set_yticks([0, 1], ["anchor wrong", "anchor restored"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-0.6, 1.02)
    ax.set_xlabel(f"span BERTScore of the repair ({method_name}, {level}% damage)")
    ax.minorticks_off()
    ax.tick_params(top=False, right=False, left=False)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _svg(fig)
