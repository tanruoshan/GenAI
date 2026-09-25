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
