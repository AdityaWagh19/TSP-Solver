"""Exact algorithms: Brute Force O(n!), Held-Karp DP O(n^2 2^n), Branch & Bound."""
import itertools
import math
import numpy as np


def brute_force(D):
    """Try all (n-1)! permutations with city 0 fixed as the start."""
    n = len(D)
    best_len, best_tour = math.inf, None
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        length = D[0, perm[0]] + D[perm[-1], 0]
        for i in range(len(perm) - 1):
            length += D[perm[i], perm[i + 1]]
        if length < best_len:
            best_len, best_tour = length, tour
    return float(best_len), list(best_tour)


def held_karp(D):
    """Dynamic programming over subsets (bitmask).
    dp[S][j] = shortest path that starts at city 0, visits exactly the cities
    in S (cities 1..n-1 mapped to bits 0..n-2) and ends at j in S.
    Recurrence: dp[S][j] = min over i in S\\{j} of dp[S\\{j}][i] + d(i, j).
    Time O(n^2 * 2^n), space O(n * 2^n). The min over i is vectorised in NumPy."""
    n = len(D)
    m = n - 1
    full = 1 << m
    dp = np.full((full, m), np.inf)
    parent = np.full((full, m), -1, dtype=np.int16)
    Dm = D[1:, 1:]
    for j in range(m):
        dp[1 << j][j] = D[0, j + 1]
    for S in range(1, full):
        if S & (S - 1) == 0:                      # singleton, already set
            continue
        for j in range(m):
            if not (S >> j) & 1:
                continue
            prev = S ^ (1 << j)
            cand = dp[prev] + Dm[:, j]            # inf where i not in prev
            i = int(np.argmin(cand))
            dp[S][j] = cand[i]
            parent[S][j] = i
    totals = dp[full - 1] + D[1:, 0]
    j = int(np.argmin(totals))
    best = float(totals[j])
    tour, S = [], full - 1
    while j != -1:
        tour.append(j + 1)
        pj = int(parent[S][j])
        S ^= (1 << j)
        j = pj
    tour.append(0)
    tour.reverse()
    return best, tour


def branch_and_bound(D, initial_bound=math.inf):
    """DFS branch & bound with a one-tree style lower bound.
    For a partial path ending at c with unvisited set U, any completion must
    leave c towards some city in U, and leave every u in U towards a city in
    (U + {0}) other than u. So
        LB = cost_so_far + min_{u in U} d(c,u) + sum_{u in U} min_{w in U+{0}, w!=u} d(u,w)
    Prune the branch when LB >= best known tour length.
    Returns (length, tour, number_of_nodes_expanded)."""
    n = len(D)
    Dl = D.tolist()
    best = {"len": initial_bound, "tour": None}
    nodes = {"count": 0}

    def dfs(path, unvisited, cost):
        nodes["count"] += 1
        cur = path[-1]
        if not unvisited:
            total = cost + Dl[cur][0]
            if total < best["len"]:
                best["len"], best["tour"] = total, path[:]
            return
        targets = unvisited + [0]
        lb = cost + min(Dl[cur][u] for u in unvisited)
        for u in unvisited:
            lb += min(Dl[u][w] for w in targets if w != u)
        if lb >= best["len"]:
            return
        # explore nearest cities first -> a good incumbent is found early
        for nxt in sorted(unvisited, key=lambda c: Dl[cur][c]):
            rest = [c for c in unvisited if c != nxt]
            path.append(nxt)
            dfs(path, rest, cost + Dl[cur][nxt])
            path.pop()

    dfs([0], list(range(1, n)), 0.0)
    return float(best["len"]), best["tour"], nodes["count"]
