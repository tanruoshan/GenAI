"""The demo's figures, drawn with matplotlib in the style of the paper's Figure 2 (SciencePlots "science" style,
the paper's font, Okabe-Ito colours and one marker per method), returned as SVG text. Every variant the page can
show is drawn here, so the page only switches between finished figures. Fonts are larger than in the paper
because the figures are shown on a projector."""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

try:
    import scienceplots  # noqa: F401  (registers the "science" styles)
    STYLE = ["science", "no-latex"]
except ImportError:  # the figures still build, with matplotlib's default style
    STYLE = ["default"]

FONTS = Path(__file__).resolve().parent / "fonts"
for f in FONTS.glob("texgyretermes-*.otf"):  # the paper's text font (TeX Gyre Termes = Nimbus Roman)
    font_manager.fontManager.addfont(str(f))
SERIF = "TeX Gyre Termes"

INK, MUTED = "#0b0b0b", "#52514e"
LEVELS = ["1w", "10", "25", "50", "75"]
TICKS = ["1 word", "10%", "25%", "50%", "75%"]
# as in scripts/paper_results.py
MAIN = [("bertdict", "BERT + dict.", "#0072B2", "o"), ("llm", "LLM few-shot", "#D55E00", "s"),
        ("lookup", "Lookup", "#E69F00", "^"), ("bart", "BART", "#009E73", "D")]
METRICS = {"frr": ("Fact Recovery Rate", True), "anchor": ("Anchor recovery", True),
           "bs": ("Span BERTScore", False), "exact": ("Exact words", True),
           "cer": ("Span CER after repair (lower is better)", True)}
PAIRS = {"H2": ("bertdict", "llm", "BERT + dict. vs. LLM"), "H2b": ("llm", "lookup", "LLM vs. lookup")}
SHORT = {"bertdict": "BERT + dict.", "llm": "LLM", "lookup": "lookup"}


def _style():
    plt.style.use(STYLE)
    plt.rcParams.update({"font.family": "serif", "font.serif": [SERIF, "Times", "DejaVu Serif"],
                         "font.size": 15, "axes.linewidth": 0.9, "svg.fonttype": "none"})


def _svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    text = buf.getvalue()
    return text[text.index("<svg"):]


def _p(p):
    if p < 0.001:
        return "p < .001"
    if p < 0.01:  # e.g. 0.0098 would round to ".010"
        return "p < .01"
    return "n.s." if p >= 0.05 else f"p = {p:.3f}".replace("0.", ".")


def metric_figure(ci, valid, tests, metric, hyp=None):
    """One metric by level for the four main methods with 95% intervals and the no-repair floor. For anchor
    recovery, Fact Recovery Rate and span BERTScore the LLM over its valid answers is dotted, as in the paper's
    Figure 2.
    With hyp ("H2" or "H2b"), the Holm-adjusted paired-test p-value of that pair is written above each level
    and the other methods are faded."""
    _style()
    title, as_pct = METRICS[metric]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(LEVELS))
    focus = PAIRS[hyp][:2] if hyp else None
    ax.plot(x, [ci[metric]["none"][lv][0] for lv in LEVELS], color=MUTED, linewidth=1.3, linestyle="--",
            label="no repair")
    for key, name, color, mark in MAIN:
        alpha = 1 if not focus or key in focus else 0.25
        v, lo, hi = (np.array([ci[metric][key][lv][i] for lv in LEVELS]) for i in range(3))
        ax.fill_between(x, lo, hi, color=color, alpha=0.15 * alpha, linewidth=0)
        ax.plot(x, v, color=color, linewidth=2.2, marker=mark, markersize=7, label=name, alpha=alpha)
    if metric in ("anchor", "frr") or (metric == "bs" and "bs" in valid["llm"]["1w"]):
        ax.plot(x, [valid["llm"][lv][metric] for lv in LEVELS], color="#D55E00", linewidth=1.6, linestyle=":",
                label="LLM, valid answers", alpha=1 if not focus or "llm" in focus else 0.25)
    if metric == "bs":
        ax.set_ylim(-0.4, 1.0)
    elif metric == "cer":
        top = max(ci[metric][k][lv][2] for k in [m[0] for m in MAIN] + ["none"] for lv in LEVELS)
        ax.set_ylim(0, round(top + 0.05, 1))
    else:
        ax.set_ylim(0, 0.8)
    if as_pct:
        ticks = ax.get_yticks()
        ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xticks(x, TICKS)
    ax.set_xlim(-0.3, 4.3)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    ax.set_title(title, loc="left", color=INK, fontsize=16)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if hyp and metric in ("anchor", "bs"):
        a, b, label = PAIRS[hyp]
        field = "mcnemar_holm" if metric == "anchor" else "wilcoxon_holm"
        # who is ahead: sentences where only one of the pair restores the anchor, or scores higher
        counts = "anchor_only" if metric == "anchor" else "bs_higher"
        top = ax.get_ylim()[1]
        y = top * 0.93 if metric == "anchor" else top - 0.08
        for i, lv in enumerate(LEVELS):
            t = next(t for t in tests if t["pair"] == [a, b] and t["level"] == lv)
            sig = t[field] < 0.05
            ax.text(i, y, _p(t[field]), ha="center", fontsize=13, style="italic", color=INK if sig else "#8a8a8a")
            if sig:
                ahead = a if t[counts][0] > t[counts][1] else b
                ax.text(i, y - (top - ax.get_ylim()[0]) * 0.055, f"{SHORT[ahead]} ahead", ha="center", fontsize=11.5,
                        color=INK)
        ax.set_title(f"{label}, Holm-adjusted {'McNemar' if metric == 'anchor' else 'Wilcoxon'} test",
                     loc="right", color=MUTED, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 3, 4] + list(range(5, len(handles))) + [0]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=len(order),
               frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=13.5, handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _svg(fig)


def strip_figure(sentences, method, level, method_name, highlight_id=None):
    """Span BERTScore of every repair of one method at one level, split by whether the anchor was restored.
    The repair of the example sentence is circled."""
    _style()
    rows = [(s["repairs"][level][method]["bs"], s["repairs"][level][method]["anchor"], s["id"]) for s in sentences]
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    rng = np.random.default_rng(0)
    # neutral ink: filled = anchor restored, hollow = anchor wrong (the method colours mean methods elsewhere)
    for ok, y, color, label in [(1, 1, "#333333", "anchor restored"), (0, 0, "#333333", "anchor wrong")]:
        pts = [(bs, sid) for bs, a, sid in rows if a == ok]
        xs = np.array([bs for bs, _ in pts])
        ys = y + rng.uniform(-0.22, 0.22, len(xs))
        ax.scatter(xs, ys, s=46, facecolors=color if ok else "none", edgecolors=color, linewidths=1.4, zorder=3)
        for (bs, sid), yy in zip(pts, ys):
            if sid == highlight_id:
                ax.scatter([bs], [yy], s=320, facecolors="none", edgecolors=INK, linewidths=2.4, zorder=4)
    n_ok = sum(a for _, a, _ in rows)
    ax.set_yticks([0, 1], [f"anchor wrong ({len(rows) - n_ok})", f"anchor restored ({n_ok})"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-0.6, 1.02)
    lv = "1 word" if level == "1w" else f"{level}%"
    ax.set_xlabel(f"span BERTScore of the repair ({method_name}, {lv} damage)")
    ax.minorticks_off()
    ax.tick_params(top=False, right=False, left=False)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return _svg(fig)


def all_figures(ci, valid, tests, sentences, talk):
    """Every figure the page can show: one per metric, the two hypothesis views of anchor recovery and span
    BERTScore, and one BERTScore strip plot per main method and level (the example sentence circled)."""
    metric = {m: metric_figure(ci, valid, tests, m) for m in METRICS}
    for hyp in PAIRS:
        metric[f"anchor_{hyp}"] = metric_figure(ci, valid, tests, "anchor", hyp)
        metric[f"bs_{hyp}"] = metric_figure(ci, valid, tests, "bs", hyp)
    strip = {f"{key}|{lv}": strip_figure(sentences, key, lv, name, talk["id"])
             for key, name, _, _ in MAIN for lv in LEVELS[1:]}
    return {"metric": metric, "strip": strip}
