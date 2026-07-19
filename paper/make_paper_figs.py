"""Paper-style figures (vector PDF, serif, compact, colorblind-safe)."""
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/niliguy/github.com/llm-d.github.io/paper/figures"
C_BASE = "#666666"   # baseline: gray
C_P2P = "#0b62a4"    # p2p: blue (colorblind-safe pair)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "grid.linewidth": 0.4,
    "grid.color": "#cccccc",
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
W1 = 3.35  # single-column width (in)


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# Fig 1: single-request crossover, gpt-oss sweep (measured 5-rep medians)
toks = np.array([2048, 8192, 16384, 32768, 49152])
rec = np.array([71, 205, 426, 983, 1695])
pull = np.array([49, 120, 196, 376, 551])
fig, ax = plt.subplots(figsize=(W1, 2.3))
ax.plot(toks / 1e3, rec, "-o", color=C_BASE, label="recompute")
ax.plot(toks / 1e3, pull, "-s", color=C_P2P, label="P2P pull")
ax.set_xlabel("prefix length (K tokens)")
ax.set_ylabel("prefill latency (ms)")
ax.set_xlim(0, 51)
despine(ax)
ax.legend(loc="upper left")
fig.savefig(f"{OUT}/crossover.pdf")
plt.close(fig)

# Fig 2: per-turn TTFT, chat multi-turn P/D (run N arm B)
turns = np.arange(8)
p50 = [1.00, 0.10, 0.13, 0.15, 0.16, 0.18, 0.20, 0.21]
p95 = [1.69, 0.12, 0.20, 0.17, 0.18, 0.21, 0.23, 0.23]
fig, ax = plt.subplots(figsize=(W1, 2.2))
ax.plot(turns, p50, "-o", color=C_P2P, label="p50")
ax.plot(turns, p95, "--s", color=C_P2P, alpha=0.55, label="p95")
ax.set_xlabel("conversation turn")
ax.set_ylabel("TTFT (s)")
ax.set_ylim(0, 1.9)
ax.set_xticks(turns)
despine(ax)
ax.legend(loc="upper right")
ax.annotate("cold prefill", (0, 1.00), textcoords="offset points",
            xytext=(6, 4), fontsize=7.5, color="#444444")
fig.savefig(f"{OUT}/turns.pdf")
plt.close(fig)

# Fig 3: paired TTFT percentiles, runs L and O (two panels)
fig, axes = plt.subplots(1, 2, figsize=(2 * W1 + 0.3, 2.3))
for ax, title, a, b in [
    (axes[0], "document Q&A (gpt-oss-120b, C=192)",
     [11.94, 71.6, 106.1], [1.16, 55.2, 80.0]),
    (axes[1], "agentic sessions (Qwen3-30B, C=16)",
     [5.22, 18.94, 30.29], [1.09, 11.77, 29.98]),
]:
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, a, w, color=C_BASE, label="P/D guide")
    ax.bar(x + w / 2, b, w, color=C_P2P, label="+ P2P")
    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95", "p99"])
    ax.set_title(title, fontsize=8)
    despine(ax)
axes[0].set_ylabel("TTFT (s)")
axes[0].legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/paired.pdf")
plt.close(fig)

print("figures written")
