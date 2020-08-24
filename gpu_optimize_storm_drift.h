#ifndef GPU_STORM_DRIFT_OPTIMIZE_INCLUDED
#define GPU_STORM_DRIFT_OPTIMIZE_INCLUDED

#include <vector>
#include <stdlib.h>

// Precision
#ifdef PRECISION_DOUBLE 
#define REAL double
#else
#define REAL float
#endif


#ifdef __cplusplus
extern "C" {
#endif

    int gpu_opt_storm_drift_initialize_2d_portable(int argc, void *argv[]);
    int gpu_opt_storm_drift_compute_2d_portable(int argc, void *argv[]);
    int gpu_opt_storm_drift_free_2d_portable(int argc, void *argv[]);
    int gpu_opt_storm_drift_initialize_3d_portable(int argc, void *argv[]);
    int gpu_opt_storm_drift_compute_3d_portable(int argc, void *argv[]);
    int gpu_opt_storm_drift_free_3d_portable(int argc, void *argv[]);

#ifdef __cplusplus
}
#endif

int gpu_opt_storm_drift_compute_2d(
    int n_coordinates,
    int n_timepoints,
    size_t n_coordinate_pairs,
    REAL gaussian_scale,
    REAL * drift_trajectory,
    REAL * output_cost_function,
    int flag_calculate_derivatives,
    REAL * output_derivatives);

int gpu_opt_storm_drift_initialize_2d(
    int n_coordinates,
    int n_time_points,
    REAL * coordinates_x,
    REAL * coordinates_y,
    int * coordinates_time,
    size_t n_coordinate_pairs,
    int * pair_indices_i,
    int * pair_indices_j);

int gpu_opt_storm_drift_free_2d();

int gpu_opt_storm_drift_compute_3d(
    int n_coordinates,
    int n_timepoints,
    size_t n_coordinate_pairs,
    REAL gaussian_scale,
    REAL * drift_trajectory,
    REAL * output_cost_function,
    int flag_calculate_derivatives,
    REAL * output_derivatives);

int gpu_opt_storm_drift_initialize_3d(
    int n_coordinates,
    int n_time_points,
    REAL * coordinates_x,
    REAL * coordinates_y,
    REAL * coordinates_z,
    int * coordinates_time,
    size_t n_coordinate_pairs,
    int * pair_indices_i,
    int * pair_indices_j);

int gpu_opt_storm_drift_free_3d();




#endif // !GPU_STORM_DRIFT_OPTIMIZE_INCLUDED