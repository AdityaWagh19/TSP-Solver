"""Instance generation and helper functions shared by all algorithms."""
import numpy as np


def random_instance(n, seed=0):
    """n random points in the unit square (Euclidean => metric TSP)."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def distance_matrix(points):
    """Full n x n Euclidean distance matrix."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def tour_length(tour, D):
    """Length of the closed tour (returns to start)."""
    return float(sum(D[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def is_valid_tour(tour, n):
    """A valid tour visits each of the n cities exactly once."""
    return len(tour) == n and sorted(tour) == list(range(n))
