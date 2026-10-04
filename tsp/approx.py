"""Approximation algorithms and heuristics for metric TSP."""
import itertools
import math


def nearest_neighbor(D, start=0):
    """Greedy heuristic, O(n^2). No constant-factor guarantee
    (worst case is Theta(log n) times optimal)."""
    n = len(D)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = tour[-1]
        nxt = min((c for c in range(n) if not visited[c]), key=lambda c: D[cur, c])
        visited[nxt] = True
        tour.append(nxt)
    return tour


def prim_mst(D):
    """Prim's algorithm, O(n^2). Returns the MST as a list of edges (u, v)."""
    n = len(D)
    in_tree = [False] * n
    key = [math.inf] * n
    parent = [-1] * n
    key[0] = 0
    edges = []
    for _ in range(n):
        u = min((i for i in range(n) if not in_tree[i]), key=lambda i: key[i])
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        for v in range(n):
            if not in_tree[v] and D[u, v] < key[v]:
                key[v], parent[v] = D[u, v], u
    return edges


def mst_double_tree(D):
    """2-approximation: MST + preorder walk (= shortcut of the doubled tree).
    Triangle inequality => tour <= 2*MST <= 2*OPT."""
    n = len(D)
    adj = {i: [] for i in range(n)}
    for u, v in prim_mst(D):
        adj[u].append(v)
        adj[v].append(u)
    tour, seen = [], set()

    def preorder(u):
        seen.add(u)
        tour.append(u)
        for v in sorted(adj[u], key=lambda x: D[u, x]):
            if v not in seen:
                preorder(v)

    preorder(0)
    return tour


def _min_weight_perfect_matching(odd, D):
    """Exact bitmask DP for <= 20 odd vertices, networkx blossom algorithm beyond."""
    k = len(odd)
    if k <= 20:
        memo = {0: (0.0, [])}

        def solve(mask):
            if mask in memo:
                return memo[mask]
            i = (mask & -mask).bit_length() - 1          # lowest unmatched vertex
            best = (math.inf, [])
            rest = mask ^ (1 << i)
            j_mask = rest
            while j_mask:
                j = (j_mask & -j_mask).bit_length() - 1
                j_mask ^= (1 << j)
                sub_cost, sub_pairs = solve(rest ^ (1 << j))
                c = D[odd[i], odd[j]] + sub_cost
                if c < best[0]:
                    best = (c, sub_pairs + [(odd[i], odd[j])])
            memo[mask] = best
            return best

        return solve((1 << k) - 1)[1]
    import networkx as nx
    G = nx.Graph()
    for a, b in itertools.combinations(odd, 2):
        G.add_edge(a, b, weight=D[a, b])
    return list(nx.min_weight_matching(G))


def christofides(D):
    """Christofides (1976): 1.5-approximation for metric TSP.
    1. MST  2. odd-degree vertices  3. min-weight perfect matching on them
    4. Eulerian circuit of MST + matching (Hierholzer)  5. shortcut repeats."""
    n = len(D)
    multigraph = {i: [] for i in range(n)}
    for u, v in prim_mst(D):
        multigraph[u].append(v)
        multigraph[v].append(u)
    odd = [v for v in range(n) if len(multigraph[v]) % 2 == 1]
    for u, v in _min_weight_perfect_matching(odd, D):
        multigraph[u].append(v)
        multigraph[v].append(u)
    adj = {u: list(vs) for u, vs in multigraph.items()}
    stack, circuit = [0], []
    while stack:                                          # Hierholzer's algorithm
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            adj[v].remove(u)
            stack.append(v)
        else:
            circuit.append(stack.pop())
    seen, tour = set(), []
    for v in circuit:                                     # shortcutting
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour


def two_opt(D, tour):
    """2-opt local search: reverse a segment whenever that shortens the tour."""
    tour = tour[:]
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour
