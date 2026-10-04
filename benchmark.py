"""Runs all experiments and writes figures + CSV/text results to outputs/."""
import csv
import math
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tsp import *
from tsp.approx import prim_mst

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
NAVY, RED, GREEN, ORANGE, GREY = "#1F3A6E", "#C0392B", "#2E8B57", "#E67E22", "#7F8C8D"
plt.rcParams.update({"font.family": "serif", "font.size": 10})


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, time.perf_counter() - t


# ---------------------------------------------------------------- Exp 1: exact runtime
def exp_exact_runtime():
    print("== Experiment 1: runtime of exact algorithms ==")
    rows, seeds = [], [0, 1, 2]
    print(f"{'n':>3} {'BruteForce(s)':>14} {'B&B(s)':>10} {'B&B nodes':>11} {'HeldKarp(s)':>12} {'(n-1)!':>12}")
    for n in range(4, 19):
        t_bf, t_bb, nodes, t_hk = [], [], [], []
        for s in seeds:
            D = distance_matrix(random_instance(n, s))
            (hk, _), dt = timed(held_karp, D)
            t_hk.append(dt)
            if n <= 10:
                (bf, _), dt = timed(brute_force, D)
                t_bf.append(dt)
                assert math.isclose(bf, hk, rel_tol=1e-9)
            if n <= 13:
                (bb, _, nd), dt = timed(branch_and_bound, D)
                t_bb.append(dt); nodes.append(nd)
                assert math.isclose(bb, hk, rel_tol=1e-9)
        r = dict(n=n,
                 bf=np.mean(t_bf) if t_bf else None,
                 bb=np.mean(t_bb) if t_bb else None,
                 nodes=np.mean(nodes) if nodes else None,
                 hk=np.mean(t_hk))
        rows.append(r)
        f = lambda x: f"{x:>10.4f}" if x is not None else f"{'-':>10}"
        print(f"{n:>3} {f(r['bf']):>14} {f(r['bb'])} {('%d' % r['nodes']) if r['nodes'] else '-':>11} "
              f"{r['hk']:>12.4f} {math.factorial(n - 1):>12.3e}")
    with open(os.path.join(OUT, "exact_runtime.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["n", "bruteforce_s", "bnb_s", "bnb_nodes", "heldkarp_s"])
        for r in rows: w.writerow([r["n"], r["bf"], r["bb"], r["nodes"], r["hk"]])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab, c, mk in [("bf", "Brute Force  O(n!)", RED, "o"),
                            ("bb", "Branch & Bound", ORANGE, "s"),
                            ("hk", "Held-Karp DP  O(n$^2$2$^n$)", NAVY, "^")]:
        pts = [(r["n"], r[key]) for r in rows if r[key]]
        ax[0].plot(*zip(*pts), marker=mk, color=c, label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlabel("number of cities n"); ax[0].set_ylabel("time (s, log scale)")
    ax[0].set_title("(a) Running time of exact algorithms"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    pts = [(r["n"], r["nodes"]) for r in rows if r["nodes"]]
    ns = [p[0] for p in pts]
    ax[1].plot(ns, [math.factorial(n - 1) for n in ns], "o--", color=RED, label="all tours (n-1)!")
    ax[1].plot(*zip(*pts), "s-", color=ORANGE, label="B&B nodes expanded")
    ax[1].set_yscale("log"); ax[1].set_xlabel("number of cities n"); ax[1].set_ylabel("search nodes (log)")
    ax[1].set_title("(b) Effect of pruning in Branch & Bound"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_exact_runtime.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Exp 2: approximation ratio
def exp_ratio(n=12, trials=40):
    print(f"\n== Experiment 2: approximation ratio vs optimum (n={n}, {trials} random instances) ==")
    algs = {"Nearest Neighbour": lambda D: nearest_neighbor(D),
            "MST 2-approx": mst_double_tree,
            "Christofides": christofides,
            "NN + 2-opt": lambda D: two_opt(D, nearest_neighbor(D))}
    ratios = {k: [] for k in algs}
    for s in range(trials):
        D = distance_matrix(random_instance(n, 100 + s))
        opt, _ = held_karp(D)
        for k, fn in algs.items():
            ratios[k].append(tour_length(fn(D), D) / opt)
    print(f"{'Algorithm':<20}{'Mean':>8}{'Median':>8}{'Worst':>8}{'Best':>8}  Theoretical bound")
    bounds = {"Nearest Neighbour": "O(log n)", "MST 2-approx": "2.0", "Christofides": "1.5", "NN + 2-opt": "none"}
    for k, v in ratios.items():
        print(f"{k:<20}{np.mean(v):>8.3f}{np.median(v):>8.3f}{max(v):>8.3f}{min(v):>8.3f}  {bounds[k]}")
    assert max(ratios["MST 2-approx"]) <= 2 and max(ratios["Christofides"]) <= 1.5
    print("Guarantee check passed: MST <= 2.0 and Christofides <= 1.5 on all instances.")
    with open(os.path.join(OUT, "approx_ratio.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["algorithm", "mean", "median", "worst", "best"])
        for k, v in ratios.items(): w.writerow([k, np.mean(v), np.median(v), max(v), min(v)])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bp = ax.boxplot(list(ratios.values()), patch_artist=True)
    ax.set_xticklabels(list(ratios.keys()))
    for patch, c in zip(bp["boxes"], [ORANGE, RED, GREEN, NAVY]):
        patch.set_facecolor(c); patch.set_alpha(.6)
    ax.axhline(1.5, ls="--", color=GREEN, lw=1); ax.text(4.45, 1.51, "1.5 (Christofides bound)", ha="right", fontsize=8, color=GREEN)
    ax.axhline(2.0, ls="--", color=RED, lw=1); ax.text(4.45, 1.99, "2.0 (MST bound)", ha="right", va="top", fontsize=8, color=RED)
    ax.axhline(1.0, color="k", lw=.8)
    ax.set_ylabel("tour length / optimal length"); ax.set_ylim(.95, 2.1)
    ax.set_title(f"Approximation ratio over {trials} random instances (n={n})")
    ax.tick_params(axis="x", labelsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ratio_boxplot.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Exp 3: larger instances
def exp_large():
    print("\n== Experiment 3: larger instances (exact methods infeasible) ==")
    print("Quality = tour length / MST weight (MST weight is a lower bound on OPT, so this over-estimates the true ratio)")
    sizes, seeds = [20, 50, 100, 150], [0, 1, 2]
    res = {k: {"q": [], "t": []} for k in ["Nearest Neighbour", "MST 2-approx", "Christofides", "NN + 2-opt"]}
    fns = {"Nearest Neighbour": lambda D: nearest_neighbor(D), "MST 2-approx": mst_double_tree,
           "Christofides": christofides, "NN + 2-opt": lambda D: two_opt(D, nearest_neighbor(D))}
    print(f"{'n':>4} " + "".join(f"{k:>34}" for k in fns))
    for n in sizes:
        q = {k: [] for k in fns}; t = {k: [] for k in fns}
        for s in seeds:
            D = distance_matrix(random_instance(n, s))
            lb = sum(D[u, v] for u, v in prim_mst(D))
            for k, fn in fns.items():
                tour, dt = timed(fn, D)
                assert is_valid_tour(tour, n)
                q[k].append(tour_length(tour, D) / lb); t[k].append(dt)
        for k in fns:
            res[k]["q"].append(np.mean(q[k])); res[k]["t"].append(np.mean(t[k]))
        print(f"{n:>4} " + "".join(f"{'ratio %.3f | %.4f s' % (np.mean(q[k]), np.mean(t[k])):>34}" for k in fns))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for (k, v), c, mk in zip(res.items(), [ORANGE, RED, GREEN, NAVY], "sod^"):
        ax[0].plot(sizes, v["q"], marker=mk, color=c, label=k)
        ax[1].plot(sizes, v["t"], marker=mk, color=c, label=k)
    ax[0].set_ylabel("tour length / MST lower bound"); ax[0].set_title("(a) Solution quality (lower is better)")
    ax[1].set_ylabel("time (s, log scale)"); ax[1].set_yscale("log"); ax[1].set_title("(b) Running time")
    ax[0].set_ylim(1.15, 1.9)
    ax[0].legend(fontsize=8, ncol=2, loc="upper center")
    ax[1].legend(fontsize=8, loc="lower right")
    for a in ax: a.set_xlabel("number of cities n"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_large_instances.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure: tours on one instance
def fig_tours(n=14, seed=3):
    pts = random_instance(n, seed); D = distance_matrix(pts)
    opt, t_opt = held_karp(D)
    mst = prim_mst(D)
    deg = np.zeros(n, int)
    for u, v in mst: deg[u] += 1; deg[v] += 1
    odd = [i for i in range(n) if deg[i] % 2]
    tours = [("Optimal (Held-Karp)", t_opt, NAVY), ("Nearest Neighbour", nearest_neighbor(D), ORANGE),
             ("MST 2-approx", mst_double_tree(D), RED), ("Christofides", christofides(D), GREEN),
             ("NN + 2-opt", two_opt(D, nearest_neighbor(D)), GREY)]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, t, c) in zip(axs.flat, tours):
        cyc = list(t) + [t[0]]
        ax.plot(pts[cyc, 0], pts[cyc, 1], "-", color=c, lw=1.8, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=45, color="white", edgecolor="k", zorder=2)
        ax.scatter(*pts[0], s=90, color="gold", edgecolor="k", zorder=3)
        L = tour_length(t, D)
        ax.set_title(f"{name}\nlength = {L:.3f}  (ratio {L / opt:.3f})", fontsize=10)
    ax = axs.flat[5]
    for u, v in mst: ax.plot(pts[[u, v], 0], pts[[u, v], 1], "-", color=NAVY, lw=1.6, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=45, color="white", edgecolor="k", zorder=2)
    ax.scatter(pts[odd, 0], pts[odd, 1], s=95, color=RED, edgecolor="k", zorder=3)
    ax.set_title(f"Minimum spanning tree (Christofides step 1)\nred = {len(odd)} odd-degree vertices", fontsize=10)
    for a in axs.flat: a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.suptitle(f"Tours produced by each algorithm on one random instance (n = {n}, gold = start city)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_tours_comparison.png"), dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Figure: Christofides pipeline
def fig_pipeline():
    steps = ["Distance\nmatrix D", "Minimum\nSpanning Tree\n(Prim)", "Find odd-degree\nvertices O\n(|O| is even)",
             "Min-weight perfect\nmatching M on O", "Eulerian circuit of\nMST + M\n(Hierholzer)", "Shortcut repeated\nvertices\n= TSP tour"]
    fig, ax = plt.subplots(figsize=(11, 2.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2)
    w, gap = 1.52, 0.28
    for i, s in enumerate(steps):
        x = i * (w + gap) + .05
        ax.add_patch(FancyBboxPatch((x, .35), w, 1.3, boxstyle="round,pad=0.03", fc="#E8EDF7", ec=NAVY, lw=1.4))
        ax.text(x + w / 2, 1.0, s, ha="center", va="center", fontsize=8.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - .02, 1.0), xytext=(x + w + .03, 1.0), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.5))
    fig.savefig(os.path.join(FIG, "fig_christofides_pipeline.png"), dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_tours()
    exp_exact_runtime()
    exp_ratio()
    exp_large()
    print("\nFigures and CSV files written to ./outputs")
