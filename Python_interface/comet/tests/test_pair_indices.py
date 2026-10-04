"""Tests for the KD-tree neighbour-pair search.

The pair list drives the whole cost function, so it is checked against a
brute-force reference on inputs small enough to enumerate exhaustively.
"""

import numpy as np
import pytest

from comet.core.pair_indices import pair_indices_kdtree


def brute_force_pairs(coords, distance):
    """All (i, j) with i < j and |coords[i] - coords[j]| <= distance."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    i, j = np.triu_indices(len(coords), k=1)
    keep = dist[i, j] <= distance
    return set(zip(i[keep].tolist(), j[keep].tolist()))


def as_pair_set(idx_i, idx_j):
    return set(zip(np.asarray(idx_i).tolist(), np.asarray(idx_j).tolist()))


class TestCorrectness:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_brute_force_3d(self, seed):
        rng = np.random.default_rng(seed)
        coords = rng.random((60, 3)) * 100.0
        distance = 25.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, distance)

        assert ok
        assert as_pair_set(idx_i, idx_j) == brute_force_pairs(coords, distance)

    @pytest.mark.parametrize("distance", [5.0, 20.0, 50.0, 200.0])
    def test_matches_brute_force_across_radii(self, distance):
        rng = np.random.default_rng(7)
        coords = rng.random((50, 3)) * 100.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, distance)

        assert ok
        assert as_pair_set(idx_i, idx_j) == brute_force_pairs(coords, distance)

    def test_known_geometry(self):
        # points on a line 10 nm apart: radius 15 links only adjacent neighbours
        coords = np.array([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
        idx_i, idx_j, ok = pair_indices_kdtree(coords, 15.0)

        assert ok
        assert as_pair_set(idx_i, idx_j) == {(0, 1), (1, 2), (2, 3)}

    def test_radius_is_inclusive(self):
        coords = np.array([[0.0, 0, 0], [10.0, 0, 0]])
        _, _, ok = pair_indices_kdtree(coords, 10.0)
        idx_i, idx_j, ok = pair_indices_kdtree(coords, 10.0)

        assert ok
        assert as_pair_set(idx_i, idx_j) == {(0, 1)}


class TestConventions:
    def test_indices_are_ordered_and_never_self_paired(self):
        rng = np.random.default_rng(11)
        coords = rng.random((80, 3)) * 50.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, 20.0)

        assert ok
        assert np.all(np.asarray(idx_i) < np.asarray(idx_j))

    def test_no_duplicate_pairs(self):
        rng = np.random.default_rng(12)
        coords = rng.random((80, 3)) * 50.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, 20.0)

        assert ok
        pairs = as_pair_set(idx_i, idx_j)
        assert len(pairs) == len(np.asarray(idx_i))

    def test_dtype_is_int32_and_contiguous(self):
        # the CUDA kernels index device arrays with int32
        rng = np.random.default_rng(13)
        coords = rng.random((40, 3)) * 50.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, 20.0)

        assert ok
        for arr in (idx_i, idx_j):
            assert arr.dtype == np.int32
            assert arr.flags["C_CONTIGUOUS"]

    def test_does_not_mutate_input(self):
        rng = np.random.default_rng(14)
        coords = rng.random((40, 3)) * 50.0
        original = coords.copy()

        pair_indices_kdtree(coords, 20.0)

        np.testing.assert_array_equal(coords, original)

    def test_estimate_pairs_does_not_mutate_input(self):
        """estimate_pairs shifts coordinates to the origin; on a caller's array
        that silently moved their data."""
        from comet.core.pair_indices import estimate_pairs

        rng = np.random.default_rng(16)
        coords = rng.random((60, 3)) * 500.0 + 1000.0
        original = coords.copy()

        estimate_pairs(coords, 50.0)

        np.testing.assert_array_equal(coords, original)

    def test_lex_floor_does_not_mutate_input(self):
        from comet.core.pair_indices import pair_indices_lex_floor_asymmetric

        rng = np.random.default_rng(17)
        coords = rng.random((60, 3)) * 500.0 + 1000.0
        original = coords.copy()

        pair_indices_lex_floor_asymmetric(coords, 50.0)

        np.testing.assert_array_equal(coords, original)


class TestEdgeCases:
    def test_no_pairs_within_radius(self):
        coords = np.array([[0.0, 0, 0], [1000.0, 0, 0], [2000.0, 0, 0]])
        idx_i, idx_j, ok = pair_indices_kdtree(coords, 10.0)

        assert ok
        assert len(idx_i) == 0 and len(idx_j) == 0

    def test_coincident_points_pair_up(self):
        coords = np.zeros((4, 3))
        idx_i, idx_j, ok = pair_indices_kdtree(coords, 1.0)

        assert ok
        # every unordered pair of the four identical points
        assert len(idx_i) == 6

    def test_two_dimensional_input(self):
        rng = np.random.default_rng(15)
        coords = rng.random((40, 2)) * 100.0
        distance = 25.0

        idx_i, idx_j, ok = pair_indices_kdtree(coords, distance)

        assert ok
        assert as_pair_set(idx_i, idx_j) == brute_force_pairs(coords, distance)


class TestSlabbedSearch:
    """The search runs in slabs along x; the slabs must be invisible in the result."""

    @staticmethod
    def reference(coords, distance):
        from scipy.spatial import cKDTree
        return set(map(tuple, cKDTree(coords).query_pairs(distance, output_type="ndarray").tolist()))

    @pytest.mark.parametrize("per_slab", [1, 7, 100, 10 ** 9])
    def test_any_number_of_slabs_finds_each_pair_once(self, monkeypatch, per_slab):
        import comet.core.pair_indices as pair_indices
        monkeypatch.setattr(pair_indices, "PAIRS_PER_SLAB", per_slab)
        rng = np.random.default_rng(5)
        coords = rng.random((3000, 3)) * np.array([2000.0, 300.0, 100.0])

        idx_i, idx_j, ok = pair_indices_kdtree(coords, 25.0)

        assert ok
        pairs = as_pair_set(idx_i, idx_j)
        assert len(pairs) == len(idx_i), "a pair was listed twice"
        assert pairs == self.reference(coords, 25.0)

    def test_points_sharing_an_x_coordinate_across_a_slab_edge(self, monkeypatch):
        import comet.core.pair_indices as pair_indices
        monkeypatch.setattr(pair_indices, "PAIRS_PER_SLAB", 3)
        rng = np.random.default_rng(6)
        coords = rng.random((2000, 2)) * 400.0
        coords[:, 0] = np.round(coords[:, 0] / 10.0) * 10.0   # many exact ties in x

        idx_i, idx_j, _ = pair_indices_kdtree(coords, 10.0)

        assert as_pair_set(idx_i, idx_j) == self.reference(coords, 10.0)
        assert len(idx_i) == len(self.reference(coords, 10.0))

    def test_an_undercounted_total_grows_rather_than_drops_pairs(self, monkeypatch):
        import comet.core.pair_indices as pair_indices
        real = pair_indices.count_pairs
        monkeypatch.setattr(pair_indices, "count_pairs", lambda c, d: max(0, real(c, d) // 2))
        rng = np.random.default_rng(7)
        coords = rng.random((1500, 3)) * 300.0

        idx_i, idx_j, _ = pair_indices_kdtree(coords, 20.0)

        assert as_pair_set(idx_i, idx_j) == self.reference(coords, 20.0)
        assert len(idx_i) == len(self.reference(coords, 20.0))

    def test_count_pairs_is_exact(self):
        from comet.core.pair_indices import count_pairs, estimate_pairs
        rng = np.random.default_rng(8)
        coords = rng.random((2500, 3)) * 500.0
        expected = len(self.reference(coords, 30.0))
        assert count_pairs(coords, 30.0) == expected
        assert estimate_pairs(coords, 30.0) == expected

    def test_tiny_inputs(self):
        for n in (0, 1):
            idx_i, idx_j, ok = pair_indices_kdtree(np.zeros((n, 3)), 1.0)
            assert ok and len(idx_i) == len(idx_j) == 0
            assert idx_i.dtype == np.int32


class TestCrowdedGeometry:
    """Points that all share (nearly) one coordinate must not defeat the slabs."""

    def test_identical_first_coordinate_is_still_exact(self, monkeypatch):
        import comet.core.pair_indices as pair_indices
        from scipy.spatial import cKDTree
        monkeypatch.setattr(pair_indices, "PAIRS_PER_SLAB", 500)
        rng = np.random.default_rng(9)
        coords = rng.normal(size=(1500, 3)) * [0.0, 5.0, 5.0]
        idx_i, idx_j, _ = pair_indices_kdtree(coords, 1.0)
        expected = set(map(tuple, cKDTree(coords).query_pairs(1.0, output_type="ndarray").tolist()))
        assert as_pair_set(idx_i, idx_j) == expected and len(idx_i) == len(expected)

    def test_all_points_within_reach_of_each_other(self, monkeypatch):
        """Every slab's halo is everything; only the bounded search path can help."""
        import tracemalloc
        import comet.core.pair_indices as pair_indices
        monkeypatch.setattr(pair_indices, "PAIRS_PER_SLAB", 20_000)
        coords = np.random.default_rng(10).normal(size=(3000, 2))
        tracemalloc.start()
        idx_i, idx_j, _ = pair_indices_kdtree(coords, 100.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        n_pairs = 3000 * 2999 // 2
        assert len(idx_i) == n_pairs
        assert as_pair_set(idx_i, idx_j) == set(zip(*np.triu_indices(3000, 1)))
        # the result is 8 B per pair; a slab may add a bounded amount, not a
        # multiple of the whole pair count
        assert peak - 8 * n_pairs < 0.5 * 8 * n_pairs
