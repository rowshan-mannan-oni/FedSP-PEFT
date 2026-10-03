#!/usr/bin/env python3
"""
Improved RQ1 per-project MAE figure.

Reads the tidy per-(seed, project, condition) MAE table
(experiments_temporal_shared/statistics_mae/results_long.csv) and draws a
grouped dot plot of per-project MAE for the three *learned* paradigms, with the
naive "predict-the-median" baseline shown as a recessive per-project reference
so the reader can see, project by project, whether the learned methods beat
doing nothing clever.

Design choices (vs. the original figure):
  * Okabe-Ito colourblind-safe hues, each paired with a distinct marker shape
    (identity is never colour-alone). Validated with the dataviz palette checker.
  * The federated method (the RQ1 subject) is the visually primary series.
  * Each project is a shaded band; the three methods sit on their own sub-row
    inside it so markers never overlap.
  * Median baseline drawn per project as a grey vertical line -> the naive floor.
  * Projects ordered by federated MAE (best at top) so the method reads top-to-bottom.
  * Test-set size n printed next to each project -> small (noisy) projects are visible.
  * Recessive grid, no chart-junk, vector PDF + raster PNG.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- data
CSV = Path("results_long.csv")   # point this at statistics_mae/results_long.csv
df = pd.read_csv(CSV)

# Test-set sizes per project (from results/summary.md; identical across conditions).
N_TEST = {
    "Apache_Mesos": 638, "Appcelerator_Studio": 619, "Aptana_Studio": 123,
    "Atlassian_Confluence_Server": 128, "DotNetNuke_Platform": 501,
    "Hyperledger_Fabric": 110, "Hyperledger_Indy_Node": 134,
    "Hyperledger_Indy_SDK": 136, "Hyperledger_Sawtooth": 190,
    "Lyrasis_Dura_Cloud": 126, "MongoDB_Core_Server": 142, "Moodle": 196,
    "Mule": 576, "Sonatype_Nexus": 295, "Spring_XD": 637,
    "The_MongoDB_Engineering": 878, "The_Titanium_SDK": 753,
    "Titanium_Mobile_Platform": 208,
}

def pretty(p):
    return p.replace("_", " ")

# aggregate over seeds (mean only; no whiskers)
agg = (df.groupby(["project", "condition"])["mae"]
         .agg(["mean"]).reset_index())

def series(cond):
    return agg[agg.condition == cond].set_index("project")

METHODS = ["Local-only", "Centralized", "FedProx"]   # the 3 markers
LABELS  = {"Local-only": "Local-only",
           "Centralized": "Centralized",
           "FedProx": "Federated (FedProx)"}
BASELINE = "Median"                                   # recessive reference

# Okabe-Ito, validated colourblind-safe. Each hue -> its own marker shape.
STYLE = {
    "Local-only":  dict(color="#0072B2", marker="o", ms=6,  z=3),   # blue circle
    "Centralized": dict(color="#009E73", marker="^", ms=6,  z=3),   # green triangle
    "FedProx":     dict(color="#D55E00", marker="D", ms=7,  z=4),   # vermillion diamond (primary)
}
GREY = "#6b6b6b"

# order projects by federated MAE (best/lowest at top)
fed = series("FedProx")["mean"]
order = list(fed.sort_values(ascending=False).index)   # worst first -> plotted bottom-up
ypos = {p: i for i, p in enumerate(order)}

# ---------------------------------------------------------------- plot
plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 13, "axes.labelsize": 11,
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
})
fig, ax = plt.subplots(figsize=(12.5, 8.0))   # wider number line, much shorter figure

means = {m: series(m) for m in METHODS}
base  = series(BASELINE)["mean"]

# Each project is a band of height 1; the three methods get their own sub-row
# inside it so markers never sit on top of each other.
OFFSET = {"Local-only": 0.24, "Centralized": 0.0, "FedProx": -0.24}
HALF   = 0.46   # half-height of a project band

for p in order:
    yc = ypos[p]
    # alternating band shading groups each project's three sub-rows
    if yc % 2 == 0:
        ax.axhspan(yc - HALF, yc + HALF, color="#f4f4f2", zorder=0)
    # per-project median baseline: a thin vertical reference spanning the band
    ax.plot([base.loc[p], base.loc[p]], [yc - HALF + 0.05, yc + HALF - 0.05],
            color=GREY, lw=1.4, zorder=2)
    # faint connector linking the project's three method markers (shows grouping)
    ax.plot([means[m].loc[p, "mean"] for m in METHODS],
            [yc + OFFSET[m] for m in METHODS],
            color="#cfcfcf", lw=1.2, zorder=1)
    for m in METHODS:
        st = STYLE[m]
        ym = yc + OFFSET[m]
        ax.plot(means[m].loc[p, "mean"], ym, marker=st["marker"], ms=st["ms"],
                color=st["color"], mec="white", mew=0.7, zorder=st["z"] + 1)

# y axis: project + n  (label centred on each band)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{pretty(p)}  (n={N_TEST[p]})" for p in order])
ax.set_ylim(-0.6, len(order) - 0.4)

# x axis
ax.set_xlabel("Mean absolute error  (lower is better)")
ax.set_xlim(0.3, 3.5)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.grid(axis="x", color="#e6e6e6", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(left=False)

ax.set_title("Per-project story-point MAE by training paradigm", pad=24, weight="bold")
ax.text(0.0, 1.018,
        "Mean over seeds 42–44 · projects ordered by federated MAE",
        transform=ax.transAxes, fontsize=9, color="#555")

# legend (methods + baseline), with shapes = secondary encoding
handles = [Line2D([0], [0], color=STYLE[m]["color"], marker=STYLE[m]["marker"],
                  ms=STYLE[m]["ms"], mec="white", mew=0.6, lw=0, label=LABELS[m])
           for m in METHODS]
handles.append(Line2D([0], [0], color=GREY, marker="|", ms=12, mew=1.6, lw=0,
                      label="Median baseline (per project)"))
ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.96,
          edgecolor="#dddddd", fontsize=9.5, handletextpad=0.5, borderpad=0.8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"results_rq1_perproject_mae_improved.{ext}",
                dpi=200, bbox_inches="tight")
print("saved results_rq1_perproject_mae_improved.pdf / .png")