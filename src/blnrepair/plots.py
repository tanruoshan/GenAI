"""Small plotting helpers for the notebooks."""
import matplotlib.pyplot as plt

INK, MUTED = "#0b0b0b", "#52514e"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#184f95"]  # light to dark: more fact tokens


def pool_bar_chart(table):
    """Grouped bars: sentences per length band, split by number of fact tokens."""
    groups = [c for c in table.columns if c not in ("total", "share")]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.26
    for j, (group, color) in enumerate(zip(groups, BLUE_RAMP)):
        xs = [i + (j - 1) * (width + 0.02) for i in range(len(table))]
        bars = ax.bar(xs, table[group], width, color=color, label=group)
        ax.bar_label(bars, fontsize=8, color=MUTED, padding=2)
    ax.set_xticks(range(len(table)), [f"{b} words" for b in table.index])
    ax.set_ylabel("sentences in the pool")
    ax.set_title("Sentence pool by length band and number of fact tokens", fontsize=10, color=INK, loc="left")
    ax.legend(title="fact tokens", frameon=False, fontsize=8, title_fontsize=8)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated pair)


def loss_curve(log, steps_per_epoch, best_epoch):
    """Fine-tuning log (runs/bert_ft_<version>_log.csv): training loss every log_every steps and
    validation loss after each epoch (epoch 0 = the pretrained model), on one axis in epochs."""
    train, val = log[log["kind"] == "train"], log[log["kind"] == "val"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(train["step"] / steps_per_epoch, train["loss"], color=TRAIN_COLOR, linewidth=2)
    ax.plot(val["epoch"], val["loss"], color=VAL_COLOR, linewidth=2, marker="o", markersize=6)
    best = val[val["epoch"] == best_epoch].iloc[0]
    ax.annotate(f"best: epoch {best_epoch}, {best['loss']:.2f}", (best["epoch"], best["loss"]), xytext=(0, -18),
                textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.text(train["step"].iloc[-1] / steps_per_epoch, train["loss"].iloc[-1], "  training", color=MUTED, fontsize=8, va="center")
    ax.text(val["epoch"].iloc[-1], val["loss"].iloc[-1], "  validation", color=MUTED, fontsize=8, va="center")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss per target piece")
    ax.set_title("Fine-tuning bert-base-cased: training and validation loss", fontsize=10, color=INK, loc="left")
    ax.set_xlim(-0.1, val["epoch"].max() + 0.8)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1 to 5 (validated palette), in fixed order


def level_lines(means, metrics, titles, order, reference="no repair"):
    """One panel per metric: level (x) against the metric's mean (y), one line per method, with a legend.
    means: a DataFrame indexed by (method, level), columns = metrics. order: every method that can appear, in
    a fixed order; a method keeps its colour (categorical slot) whether or not the others are present.
    The reference method (the damaged text itself) is drawn as a muted dashed line."""
    present = set(means.index.get_level_values(0))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for ax, metric, title in zip(axes, metrics, titles):
        if reference in present:
            series = means.loc[reference, metric]
            ax.plot(range(len(series)), series.values, color=MUTED, linewidth=1.5, linestyle="--", label=reference)
        for method, color in zip(order, SERIES):
            if method not in present:
                continue
            series = means.loc[method, metric]
            ax.plot(range(len(series)), series.values, color=color, linewidth=2, marker="o", markersize=6, label=method)
        ax.set_xticks(range(len(series)), [f"{lv}%" if lv != "1w" else "1 word" for lv in series.index])
        ax.set_xlabel("damage level (share of sentence words)")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.yaxis.grid(True, color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
