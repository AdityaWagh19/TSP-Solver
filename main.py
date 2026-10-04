"""Demo: run every algorithm on one random Euclidean instance and compare."""
import sys
import time
from tsp import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
D = distance_matrix(random_instance(n, seed))
print(f"TSP demo | cities = {n} | seed = {seed}\n")

rows = []


def run(name, fn, kind):
    t0 = time.perf_counter()
    res = fn()
    dt = (time.perf_counter() - t0) * 1000
    tour = res[1] if isinstance(res, tuple) else res
    length = tour_length(tour, D)
    assert is_valid_tour(tour, n)
    rows.append((name, kind, length, dt, tour))


run("Brute Force", lambda: brute_force(D), "exact") if n <= 10 else None
run("Held-Karp DP", lambda: held_karp(D), "exact")
run("Branch & Bound", lambda: branch_and_bound(D), "exact") if n <= 13 else None
run("Nearest Neighbour", lambda: nearest_neighbor(D), "heuristic")
run("MST 2-approx", lambda: mst_double_tree(D), "2-approx")
run("Christofides", lambda: christofides(D), "1.5-approx")
run("NN + 2-opt", lambda: two_opt(D, nearest_neighbor(D)), "heuristic")

opt = min(r[2] for r in rows if r[1] == "exact")
print(f"{'Algorithm':<20}{'Type':<12}{'Length':>9}{'Ratio':>8}{'Time(ms)':>11}")
print("-" * 60)
for name, kind, length, dt, tour in rows:
    print(f"{name:<20}{kind:<12}{length:>9.4f}{length / opt:>8.3f}{dt:>11.2f}")
print("\nOptimal tour (Held-Karp):", [r[4] for r in rows if r[0] == "Held-Karp DP"][0])
