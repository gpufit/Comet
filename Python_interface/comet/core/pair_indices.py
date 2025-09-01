import numpy as np
from scipy.spatial import cKDTree
import h5py


def pair_indices_kdtree(coordinates, distance):
    """
    Find all pairs of points within a certain distance using a KD-tree.
    Parameters:
    - coordinates: np.ndarray of shape (N, D) where N is the number of points and D is the dimensionality.
    - distance: float, the maximum distance to consider points as a pair.
    Returns:
    - idx1: np.ndarray of shape (M,), indices of the first point in each pair.
    - idx2: np.ndarray of shape (M,), indices of the second point in each pair.
    """
    tree = cKDTree(coordinates)
    while True:
        try:
            pairs = tree.query_pairs(r=distance, output_type='ndarray')
            break
        except MemoryError:
            distance *= 0.8
            print(f"[pair_indices_kdtree] Reducing distance to {distance:.2f} due to memory error.")

    print(f"[pair_indices_kdtree] Found {len(pairs):,} pairs")
    return np.ascontiguousarray(pairs[:, 0], dtype=np.int32), np.ascontiguousarray(pairs[:, 1], dtype=np.int32)


def pair_indices_kdtree_full_to_file_recursion(coordinates, distance, filename=None, split_dimension=0, indices=None):
    """
    Recursively find all pairs of points within a certain distance using a KD-tree,
    and save the results to an HDF5 file to avoid memory issues.
    Parameters:
    - coordinates: np.ndarray of shape (N, D) where N is the number of points and D is the dimensionality.
    - distance: float, the maximum distance to consider points as a pair.
    - filename: str, path to the HDF5 file where results will be saved.
    - split_dimension: int, the dimension along which to split the data when a MemoryError occurs.
    - indices: np.ndarray of shape (N,), original indices of the points in the full dataset.
    Returns:
    - None (results are saved to the specified HDF5 file)
    """
    if indices is None:
        indices = np.arange(len(coordinates))

    try:
        tree = cKDTree(coordinates)
        pairs = tree.query_pairs(r=distance, output_type='ndarray')
        global_pairs = np.stack([indices[pairs[:, 0]], indices[pairs[:, 1]]], axis=1)

        with h5py.File(filename, 'a') as f:
            grp_name = f"pair_indices_{len(f.keys()):04d}"
            f.create_dataset(grp_name, data=global_pairs, compression="gzip")
            print(f"[pair_indices_kdtree_full_to_file_recursion] Saved {len(global_pairs):,} pairs to '{grp_name}'")

    except MemoryError:
        print("[pair_indices_kdtree_full_to_file_recursion] MemoryError - splitting...")
        median_val = np.median(coordinates[:, split_dimension])
        left_mask = coordinates[:, split_dimension] < median_val
        right_mask = ~left_mask
        dim_next = (split_dimension + 1) % coordinates.shape[1]

        pair_indices_kdtree_full_to_file_recursion(
            coordinates[left_mask], distance, filename,
            split_dimension=dim_next, indices=indices[left_mask]
        )
        pair_indices_kdtree_full_to_file_recursion(
            coordinates[right_mask], distance, filename,
            split_dimension=dim_next, indices=indices[right_mask]
        )


