import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import numba
import time
import math


def pair_indices_lex_dblchk(coordinates, distance):
    # t = time.time()
    idx_i, idx_j = pair_indices_lex(coordinates, distance)

    idx_i_2, idx_j_2 = pair_indices_lex(np.asarray(coordinates + (0.51 * distance)),
                                        distance)
    # print(f"time of 2 lexes: {time.time() - t}")
    # t = time.time()
    idx_i_flat = np.concatenate((np.concatenate(idx_i), np.concatenate(idx_i_2)))
    idx_j_flat = np.concatenate((np.concatenate(idx_j), np.concatenate(idx_j_2)))
    pair_idc = np.concatenate((idx_i_flat, idx_j_flat)).reshape((2, len(idx_i_flat)))
    # print(f"concat time: {time.time() - t}")
    # pair_idc = np.unique(pair_idc, axis=1) # very slow
    _, idx = np.unique(np.ascontiguousarray(pair_idc
                                            ).view(np.dtype((np.void,
                                                             pair_idc.dtype.itemsize * pair_idc.shape[0]))),
                       return_index=True)
    pair_idc = pair_idc[:, idx]
    # sort_idc = np.lexsort(np.array(list(map(tuple, pair_idc.T))).T)
    # unique_pair_idc = np.unique(pair_idc.T[sort_idc])
    idx_i_final = pair_idc[0, :]
    idx_j_final = pair_idc[1, :]
    # print(f"time of uniquing: {time.time() - t}")
    return idx_i_final, idx_j_final


def pair_indices_lex_floor(coordinates, distance):
    for i in range(len(coordinates[0])):
        coordinates[:, i] -= np.min(coordinates[:, i])
    coordinates = np.array(np.floor(coordinates / distance), dtype=int)
    coordinates = np.array(list(map(tuple, coordinates)))

    sort_indices = np.lexsort(coordinates.T)

    # get the unique tuples and their counts
    unique_tuples, counts = np.unique(coordinates[sort_indices], axis=0, return_counts=True)

    # get the indices of the similar tuples
    similar_indices = np.split(sort_indices, np.cumsum(counts[:-1]))

    idx_i = []
    idx_j = []
    for i in range(len(similar_indices)):
        if len(similar_indices[i]) > 1:
            idc = similar_indices[i]
            for j in range(len(idc)-1):
                tmp_n_entries = (len(idc) - j) - 1
                idx_i.append(np.repeat(np.int32(idc[j]), tmp_n_entries))
                idx_j.append(idc[np.arange(tmp_n_entries) + (j+1)])
    return idx_i, idx_j


def pair_indices_lex(coordinates, distance):
    coordinates = np.array(np.round(coordinates / distance, 0), dtype=int)
    coordinates = np.array(list(map(tuple, coordinates)))

    sort_indices = np.lexsort(coordinates.T)

    # get the unique tuples and their counts
    unique_tuples, counts = np.unique(coordinates[sort_indices], axis=0, return_counts=True)

    # get the indices of the similar tuples
    similar_indices = np.split(sort_indices, np.cumsum(counts[:-1]))

    idx_i = []
    idx_j = []
    for i in range(len(similar_indices)):
        if len(similar_indices[i]) > 1:
            idc = similar_indices[i]
            for j in range(len(idc)):
                idx_j.append(idc[np.arange(len(idc)) > j])
                idx_i.append(np.ones_like(idx_j[-1])*idc[j])
    return idx_i, idx_j


def pair_indices_digitize(coordinates, distance):
    bins_x = np.linspace(np.min(coordinates[:, 0]), np.max(coordinates[:, 0]), int(np.ceil((np.max(coordinates[:, 0])-np.min(coordinates[:, 0]))/distance)))
    bins_y = np.linspace(np.min(coordinates[:, 1]), np.max(coordinates[:, 1]), int(np.ceil((np.max(coordinates[:, 1])-np.min(coordinates[:, 1]))/distance)))
    bins = [bins_x, bins_y]
    digitized = []
    for i in range(len(bins)):
        digitized.append(np.digitize(coordinates[:, i], bins[i], right=False))
    digitized = np.concatenate(digitized).reshape(len(coordinates), 2) - 1
    idx_i = []
    idx_j = []
    for i in range(len(bins_x)):
        for j in range(len(bins_y)):
            coords_in_bin = []
            for k in range(len(digitized)):
                if digitized[k, 0] == i and digitized[k, 1] == j:
                    coords_in_bin.append(k)
            coords_in_bin = np.asarray(coords_in_bin)
            for k in range(len(coords_in_bin)):
                idx_j.append(coords_in_bin[np.arange(len(coords_in_bin)) != k])
                idx_i.append(np.ones(len(idx_j[-1]))*coords_in_bin[k])
    return idx_i, idx_j


def pair_indices_simplified(coordinates, distance, dim='2D'):
    coordinates = np.array(np.round(coordinates/distance, 0), dtype=int)
    idx_i = []
    idx_j = []
    if dim == '2D':
        for i in range(len(coordinates)):
            x_overlap = np.where(coordinates[i, 0] == coordinates[np.arange(len(coordinates)), 0])[0]
            y_overlap = np.where(coordinates[i, 1] == coordinates[np.arange(len(coordinates)), 1])[0]
            overlap = np.intersect1d(x_overlap, y_overlap)
            overlap = np.delete(overlap, np.where(overlap == i)[0])
            if not overlap.size == 0:
                idx_j.append(overlap)
                idx_i.append(i*np.ones_like(idx_j[-1]))
    else:
        pass
    return idx_i, idx_j


@numba.jit
def pair_indices_simplified_vectorized(coordinates, distance):
    # somehow takes longer...
    np.round(coordinates/distance, 0, coordinates)
    idx_i = []
    idx_j = []
    for i in range(len(coordinates)):
        x_overlap = np.where(coordinates[i, 0] == coordinates[np.arange(len(coordinates)), 0])[0]
        y_overlap = np.where(coordinates[i, 1] == coordinates[np.arange(len(coordinates)), 1])[0]
        overlap = np.intersect1d(x_overlap, y_overlap)
        overlap = np.delete(overlap, np.where(overlap == i)[0])
        if not overlap.size == 0:
            idx_j.append(overlap)
            idx_i.append(i * np.ones_like(idx_j[-1]))
    return idx_i, idx_j


def pair_indices(coordinates, distance):
    # to be tested
    idx_i = []
    idx_j = []
    for i in range(len(coordinates)):
        overlap = np.where(np.sum((coordinates[i] - coordinates[np.arange(len(coordinates))]) ** 2, 1) ** 2
                           < distance ** 2)[0]
        overlap = np.delete(overlap, np.where(overlap == i)[0])
        idx_j.append(overlap)
        idx_i.append(np.ones_like(idx_j[-1]) * i)
    return idx_i, idx_j


@cuda.jit
def pair_indices_gpu_2d(d_pair_idc, n_pair_indices, d_distance_sq=np.float64,  d_coordinates=np.array([[]])):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < n_pair_indices:
        if ((d_coordinates[d_pair_idc[pos, 0], 0] - d_coordinates[d_pair_idc[pos, 1], 0])**2+(d_coordinates[d_pair_idc[pos, 0], 1] - d_coordinates[d_pair_idc[pos, 1], 1])**2) > d_distance_sq:
            d_pair_idc[pos] = -1


def cuda_wrapper_pair_indices(coordinates, distance):
    """ is definitely faster for lots of pair indices, but can run out of memory at some point, because we have to pass
    # the whole array of possible pair indices to the device... maybe a future solution could split the task in
    # multiple steps """
    # create an array with all possible pair indices
    n_coords = len(coordinates)
    pair_idc = np.zeros((int(.5*n_coords*(n_coords-1)), 2))
    idx = 0
    for i in range(n_coords-1):
        tmp = np.dstack((np.ones(n_coords-i-1)*i, np.arange(n_coords-i-1)+i+1))[0]
        pair_idc[idx:idx+n_coords-i-1] = tmp
        idx = idx+n_coords-i-1
    # send arrays to gpu and check every pair in parallel if it's invalid (=distance of points to big) and set them to
    # [-1 -1]
    d_pair_idc = cuda.to_device(np.asarray(pair_idc, dtype=np.int64))
    d_coordinates = cuda.to_device(np.asarray(coordinates, dtype=np.float64))
    n_pair_indices = len(pair_idc)
    threadsperblock = 256
    blockspergrid = (len(coordinates) + (threadsperblock - 1)) // threadsperblock
    pair_indices_gpu_2d[blockspergrid, threadsperblock](d_pair_idc, n_pair_indices, distance**2, d_coordinates)
    # copy everything back to cpu, throw out all the invalid ones and reshape the pair indices correctly
    pair_idc = d_pair_idc.copy_to_host()
    pair_idc = pair_idc[pair_idc != -1]
    pair_idc = pair_idc.reshape(int(len(pair_idc)/2), 2)
    idx_i = pair_idc[:, 0]
    idx_j = pair_idc[:, 1]
    return idx_i, idx_j


def main():
    N = 5000
    a = np.random.rand(N, 3)*1

    #a = np.asarray([[0,0],[0,0],[0,.55]])

    dist = 1
    t = time.time()
    pair_indices_lex_dblchk(a, dist)
    print(f"dbl lex {time.time() - t}")
    t = time.time()
    idx_i_standard, idx_j_standard = pair_indices(a, dist)
    print(f"standard {time.time() - t}")
    t = time.time()
    idx_i_lex, idx_j_lex = pair_indices_lex(a, dist)
    print(f"lex {time.time() - t}")
    t = time.time()
    idx_i_simplf, idx_j_simplf = pair_indices_simplified(a, dist)
    print(f"simplified {time.time() - t}")
    t = time.time()
    pair_indices_simplified_vectorized(a, dist)
    print(f"simplified vectorized{time.time() - t}")
    t = time.time()
    #cuda_wrapper_pair_indices(a, dist)
    print(f"cuda {time.time() - t}")
    t = time.time()
    idx_i_digi, idx_j_digi = pair_indices_digitize(a, dist)
    print(f"digitize {time.time() - t}")


if __name__ == "__main__":
    main()
    ####test_pair_idc_simpl()
