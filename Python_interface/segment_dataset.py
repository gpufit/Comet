import numpy as np

"""simple scripts to segments smlm data. The main assumption here is that locs is a 3D or 4D array with the following
 shape: 
 
    if 2D dataset: locs.shape = (n_locs, 3) # X,Y,Frame
    
    if 3D dataset: locs.shape = (n_locs, 3) # X,Y,Z,Frame
    
 """


def segment_by_num_windows(locs_nm, n_windows, return_n_segments=False):
    n_windows = int(n_windows)
    n_locs = len(locs_nm[:, 0])
    n_locs_per_window = int(np.ceil(n_locs/n_windows))
    if return_n_segments:
        return segment_by_num_locs_per_window(locs_nm, n_locs_per_window), n_windows
    else:
        return segment_by_num_locs_per_window(locs_nm, n_locs_per_window)


def segment_by_num_locs_per_window(locs_nm, n_locs_per_window, return_n_segments=False):
    locs_nm[:, -1] = 0
    n_locs_per_window = int(n_locs_per_window)
    n_locs = len(locs_nm[:, 0])
    n_windows = int(np.ceil(n_locs/n_locs_per_window))
    for i in range(n_windows-1):
        locs_nm[(i+1)*n_locs_per_window:, -1] += 1
    if return_n_segments:
        return locs_nm, n_windows
    else:
        return locs_nm


def segment_by_frame_windows(locs_nm, n_frames_per_seg, return_n_segments=False):
    n_frames = int(np.max(locs_nm[:, -1]) - np.min(locs_nm[:, -1]))
    n_segments = int(np.ceil(n_frames/n_frames_per_seg))
    locs_nm[:, -1] = np.floor((locs_nm[:, -1] - np.min(locs_nm[:, -1]))/n_frames_per_seg)
    if return_n_segments:
        return locs_nm, n_segments
    else:
        return locs_nm


def main():
    N = 20
    test_data = np.zeros((N, 4))
    test_data[:, :-1] = np.random.random((N, 3))
    test_data[:, -1] = np.random.poisson(10, N)
    seg_locs_1 = segment_by_num_locs_per_window(test_data, N/10)
    seg_locs_2 = segment_by_frame_windows(test_data, N / 10)
    seg_locs_3 = segment_by_num_windows(test_data, N / 10)
    print("done")


if __name__ == "__main__":
    main()