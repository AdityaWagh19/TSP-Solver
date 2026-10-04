import math
import pytest
from tsp import *


@pytest.mark.parametrize("seed", range(5))
def test_exact_algorithms_agree(seed):
    D = distance_matrix(random_instance(8, seed))
    bf, _ = brute_force(D)
    hk, t_hk = held_karp(D)
    bb, t_bb, _ = branch_and_bound(D)
    assert math.isclose(bf, hk, rel_tol=1e-9) and math.isclose(bf, bb, rel_tol=1e-9)
    assert is_valid_tour(t_hk, 8) and is_valid_tour(t_bb, 8)
    assert math.isclose(tour_length(t_hk, D), hk, rel_tol=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_approximation_guarantees(seed):
    n = 11
    D = distance_matrix(random_instance(n, seed))
    opt, _ = held_karp(D)
    for fn, bound in [(mst_double_tree, 2.0), (christofides, 1.5)]:
        t = fn(D)
        assert is_valid_tour(t, n)
        assert tour_length(t, D) <= bound * opt + 1e-9


def test_nn_and_2opt_valid_and_improving():
    D = distance_matrix(random_instance(30, 1))
    t = nearest_neighbor(D)
    t2 = two_opt(D, t)
    assert is_valid_tour(t, 30) and is_valid_tour(t2, 30)
    assert tour_length(t2, D) <= tour_length(t, D) + 1e-12
