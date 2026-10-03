#!/usr/bin/env python3
"""
Critical-difference (Demsar) diagram for the RQ1 MAE ranking.

Reads the Friedman average ranks (friedman.json) and the Nemenyi pairwise
p-value matrix (nemenyi.csv) and draws a clean CD diagram:

  * rank scale along the top (best rank = 1 on the right);
  * each method on its OWN label row, the better half labelled to the right and
    the worse half to the left, so labels never share space or cross the axis;
  * horizontal bars link methods that are NOT significantly different
    (maximal cliques of Nemenyi p > alpha).

This layout is what fixes the overlap in the original figure: there every label
sat below the axis with a drop-line, and with five of six ranks bunched between
3 and 4.4 they collided. Here each method gets a dedicated row.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- data
FRIEDMAN = json.loads(Path("friedman.json").read_text())      # average_ranks, etc.
PMAT     = pd.read_csv("nemenyi.csv", index_col=0)            # pairwise p-values
ALPHA    = 0.05

ranks   = FRIEDMAN["average_ranks"]
methods = sorted(ranks, key=ranks.get)          # best (lowest rank) first
avg     = {m: ranks[m] for m in methods}
k       = len(methods)
LOWV, HIGHV = 1, k                               # rank 1 .. k

# ---- maximal cliques of "not significantly different" methods (p > alpha) ----
adj = {m: set() for m in methods}
for a, b in combinations(methods, 2):
    if PMAT.loc[a, b] > ALPHA:
        adj[a].add(b)
        adj[b].add(a)

def bron_kerbosch(R, P, X, out):
    if not P and not X:
        out.append(R)
        return
    for v in list(P):
        bron_kerbosch(R | {v}, P & adj[v], X & adj[v], out)
        P = P - {v}
        X = X | {v}

_cliques = []
bron_kerbosch(set(), set(methods), set(), _cliques)
# each bar = (min_rank, max_rank) of a maximal clique with >= 2 members
bars = sorted({(min(avg[m] for m in c), max(avg[m] for m in c))
               for c in _cliques if len(c) >= 2})

# ---------------------------------------------------------------- geometry
# plot coordinate: xpos = HIGHV - rank  ->  rank 1 sits on the right
def xpos(rank):
    return HIGHV - rank

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9.0, 3.4))
ax.set_xlim(-0.6, (HIGHV - LOWV) + 0.6)
ax.set_ylim(-1.15, 0.55)
ax.axis("off")

AXIS_Y   = 0.0
TICK_UP  = 0.09
LABEL_Y0 = -0.30      # first (shallowest) label row
LABEL_DY = -0.19      # row-to-row step
EDGE_PAD = 0.45       # how far past the end ticks the labels sit
BAR_Y0   = -0.10      # first clique bar, just under the axis
BAR_DY   = -0.075

# --- rank axis + ticks (numbers along the top) ---
ax.plot([xpos(HIGHV), xpos(LOWV)], [AXIS_Y, AXIS_Y], color="black", lw=1.4, zorder=3)
for r in range(LOWV, HIGHV + 1):
    x = xpos(r)
    ax.plot([x, x], [AXIS_Y, AXIS_Y + TICK_UP], color="black", lw=1.1, zorder=3)
    ax.text(x, AXIS_Y + TICK_UP + 0.04, str(r), ha="center", va="bottom", fontsize=11)

# --- clique bars (methods not significantly different) ---
for i, (rmin, rmax) in enumerate(bars):
    y = BAR_Y0 + i * BAR_DY
    ax.plot([xpos(rmax), xpos(rmin)], [y, y],
            color="#c0392b", lw=4.2, solid_capstyle="round", zorder=2)

# --- method connector lines + labels, split left / right ---
half  = (k + 1) // 2
right = methods[:half]            # best ranks -> labelled on the right
left  = methods[half:]            # worst ranks -> labelled on the left
x_right = xpos(LOWV) + EDGE_PAD    # right label anchor
x_left  = xpos(HIGHV) - EDGE_PAD   # left label anchor

def place(side_methods, anchor_x, ha):
    for i, m in enumerate(side_methods):
        x = xpos(avg[m])
        y = LABEL_Y0 + i * LABEL_DY
        ax.plot([x, x], [AXIS_Y, y], color="0.35", lw=1.0, zorder=1)   # vertical drop
        ax.plot([x, anchor_x], [y, y], color="0.35", lw=1.0, zorder=1) # horizontal to edge
        ax.text(anchor_x + (0.06 if ha == "left" else -0.06), y,
                f"{m} ({avg[m]:.2f})", ha=ha, va="center", fontsize=11)

place(right, x_right, ha="left")                 # Local-only, Centralized, Median ...
place(list(reversed(left)), x_left, ha="right")  # worst nearest the left end, shallowest

ax.set_title(
    "Critical-difference diagram for MAE  (lower rank = better; "
    "bars link methods not\nsignificantly different, Nemenyi $p > 0.05$)",
    fontsize=11, pad=10)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"results_rq1_cd_mae_improved.{ext}", dpi=200, bbox_inches="tight")
print("saved; bars =", bars)