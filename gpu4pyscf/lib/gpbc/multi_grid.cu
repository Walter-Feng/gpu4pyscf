#include <cstdio>
#include <cuda_runtime.h>
#include <gint/config.h>
#include <gint/gint.h>
#include <cassert>
#include <gint/cuda_alloc.cuh>

#define atm(SLOT, I)     atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I)     bas[BAS_SLOTS * (I) + (SLOT)]

#define EIJ_CUTOFF       60

__host__ __device__ double distance_squared(const double x, const double y, const double z) {
    return x * x + y * y + z * z;
}


template<int ANG>
__device__
static void _cart2sph(double g_cart[15], double *g_sph, int stride, int grid_id) {
    if (ANG == 0) {
        g_sph[grid_id] += 0.282094791773878143 * g_cart[0];
    } else if (ANG == 1) {
        g_sph[grid_id] += 0.488602511902919921 * g_cart[0];
        g_sph[grid_id + stride] += 0.488602511902919921 * g_cart[1];
        g_sph[grid_id + 2 * stride] += 0.488602511902919921 * g_cart[2];
    } else if (ANG == 2) {
        g_sph[grid_id] += 1.092548430592079070 * g_cart[1];
        g_sph[grid_id + stride] += 1.092548430592079070 * g_cart[4];
        g_sph[grid_id + 2 * stride] +=
                0.630783130505040012 * g_cart[5] - 0.315391565252520002 * (g_cart[0] + g_cart[3]);
        g_sph[grid_id + 3 * stride] += 1.092548430592079070 * g_cart[2];
        g_sph[grid_id + 4 * stride] += 0.546274215296039535 * (g_cart[0] - g_cart[3]);
    } else if (ANG == 3) {
        g_sph[grid_id] += 1.770130769779930531 * g_cart[1] - 0.590043589926643510 * g_cart[6];
        g_sph[grid_id + stride] += 2.890611442640554055 * g_cart[4];
        g_sph[grid_id + 2 * stride] +=
                1.828183197857862944 * g_cart[8] - 0.457045799464465739 * (g_cart[1] + g_cart[6]);
        g_sph[grid_id + 3 * stride] +=
                0.746352665180230782 * g_cart[9] - 1.119528997770346170 * (g_cart[2] + g_cart[7]);
        g_sph[grid_id + 4 * stride] +=
                1.828183197857862944 * g_cart[5] - 0.457045799464465739 * (g_cart[0] + g_cart[3]);
        g_sph[grid_id + 5 * stride] += 1.445305721320277020 * (g_cart[2] - g_cart[7]);
        g_sph[grid_id + 6 * stride] += 0.590043589926643510 * g_cart[0] - 1.770130769779930530 * g_cart[3];
    } else if (ANG == 4) {
        g_sph[grid_id] += 2.503342941796704538 * (g_cart[1] - g_cart[6]);
        g_sph[grid_id + stride] += 5.310392309339791593 * g_cart[4] - 1.770130769779930530 * g_cart[11];
        g_sph[grid_id + 2 * stride] +=
                5.677048174545360108 * g_cart[8] - 0.946174695757560014 * (g_cart[1] + g_cart[6]);
        g_sph[grid_id + 3 * stride] +=
                2.676186174229156671 * g_cart[13] - 2.007139630671867500 * (g_cart[4] + g_cart[11]);
        g_sph[grid_id + 4 * stride] +=
                0.317356640745612911 * (g_cart[0] + g_cart[10]) + 0.634713281491225822 * g_cart[3] -
                2.538853125964903290 * (g_cart[5] + g_cart[12]) + 0.846284375321634430 * g_cart[14];
        g_sph[grid_id + 5 * stride] +=
                2.676186174229156671 * g_cart[9] - 2.007139630671867500 * (g_cart[2] + g_cart[7]);
        g_sph[grid_id + 6 * stride] +=
                2.838524087272680054 * (g_cart[5] - g_cart[12]) + 0.473087347878780009 * (g_cart[10] - g_cart[0]);
        g_sph[grid_id + 7 * stride] += 1.770130769779930531 * g_cart[2] - 5.310392309339791590 * g_cart[7];
        g_sph[grid_id + 8 * stride] +=
                0.625835735449176134 * (g_cart[0] + g_cart[10]) - 3.755014412695056800 * g_cart[3];
    }
}

template<int ANG>
__device__
static void _memset_cart(double *g_cart, int stride, int grid_id) {
    if (ANG == 0) {
        g_cart[grid_id] = 0.0;
    } else if (ANG == 1) {
        g_cart[grid_id] = 0.0;
        g_cart[grid_id + stride] = 0.0;
        g_cart[grid_id + 2 * stride] = 0.0;
    } else if (ANG == 2) {
        g_cart[grid_id] = 0.0;
        g_cart[grid_id + stride] = 0.0;
        g_cart[grid_id + 2 * stride] = 0.0;
        g_cart[grid_id + 3 * stride] = 0.0;
        g_cart[grid_id + 4 * stride] = 0.0;
        g_cart[grid_id + 5 * stride] = 0.0;
    } else if (ANG == 3) {
        g_cart[grid_id] = 0.0;
        g_cart[grid_id + stride] = 0.0;
        g_cart[grid_id + 2 * stride] = 0.0;
        g_cart[grid_id + 3 * stride] = 0.0;
        g_cart[grid_id + 4 * stride] = 0.0;
        g_cart[grid_id + 5 * stride] = 0.0;
        g_cart[grid_id + 6 * stride] = 0.0;
        g_cart[grid_id + 7 * stride] = 0.0;
        g_cart[grid_id + 8 * stride] = 0.0;
        g_cart[grid_id + 9 * stride] = 0.0;
    } else if (ANG == 4) {
        g_cart[grid_id] = 0.0;
        g_cart[grid_id + stride] = 0.0;
        g_cart[grid_id + 2 * stride] = 0.0;
        g_cart[grid_id + 3 * stride] = 0.0;
        g_cart[grid_id + 4 * stride] = 0.0;
        g_cart[grid_id + 5 * stride] = 0.0;
        g_cart[grid_id + 6 * stride] = 0.0;
        g_cart[grid_id + 7 * stride] = 0.0;
        g_cart[grid_id + 8 * stride] = 0.0;
        g_cart[grid_id + 9 * stride] = 0.0;
        g_cart[grid_id + 10 * stride] = 0.0;
        g_cart[grid_id + 11 * stride] = 0.0;
        g_cart[grid_id + 12 * stride] = 0.0;
        g_cart[grid_id + 14 * stride] = 0.0;
    } else {
        int i = 0;
        for (int lx = ANG; lx >= 0; lx--) {
            for (int ly = ANG - lx; ly >= 0; ly--, i++) {
                g_cart[grid_id + i * stride] = 0.0;
            }
        }
    }
}

template<int ANG>
__device__
static void _memset_sph(double *g_sph, int stride, int grid_id) {
    if (ANG == 0) {
        g_sph[grid_id] = 0.0;
    } else if (ANG == 1) {
        g_sph[grid_id] = 0.0;
        g_sph[grid_id + stride] = 0.0;
        g_sph[grid_id + 2 * stride] = 0.0;
    } else if (ANG == 2) {
        g_sph[grid_id] = 0.0;
        g_sph[grid_id + stride] = 0.0;
        g_sph[grid_id + 2 * stride] = 0.0;
        g_sph[grid_id + 3 * stride] = 0.0;
        g_sph[grid_id + 4 * stride] = 0.0;
    } else if (ANG == 3) {
        g_sph[grid_id] = 0.0;
        g_sph[grid_id + stride] = 0.0;
        g_sph[grid_id + 2 * stride] = 0.0;
        g_sph[grid_id + 3 * stride] = 0.0;
        g_sph[grid_id + 4 * stride] = 0.0;
        g_sph[grid_id + 5 * stride] = 0.0;
        g_sph[grid_id + 6 * stride] = 0.0;
    } else if (ANG == 4) {
        g_sph[grid_id] = 0.0;
        g_sph[grid_id + stride] = 0.0;
        g_sph[grid_id + 2 * stride] = 0.0;
        g_sph[grid_id + 3 * stride] = 0.0;
        g_sph[grid_id + 4 * stride] = 0.0;
        g_sph[grid_id + 5 * stride] = 0.0;
        g_sph[grid_id + 6 * stride] = 0.0;
        g_sph[grid_id + 7 * stride] = 0.0;
        g_sph[grid_id + 8 * stride] = 0.0;
    }
}

template<int ANG>
__device__
static void _cart_gto(double *g, double fx, double fy, double fz) {
    for (int lx = ANG, i = 0; lx >= 0; lx--) {
        for (int ly = ANG - lx; ly >= 0; ly--, i++) {
            int lz = ANG - lx - ly;
            g[i] = pow(fx, lx) * pow(fy, ly) * pow(fz, lz);
        }
    }
}

template<int n_channels, int i_angular, int j_angular>
__global__ void evaluate_density_kernel(
        double *density, const double *density_matrices, const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int mesh_a, const int mesh_b, const int mesh_c,
        const int *atm, const int *bas, const double *env) {
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
    const int density_matrix_stride = n_i_functions * n_j_functions;
    const int density_matrix_channel_stride = density_matrix_stride * n_images;

    if (a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c) {
        return;
    }

    const double position_x = lattice_vectors[0] * a_index / mesh_a + lattice_vectors[3] * b_index / mesh_b +
                              lattice_vectors[6] * c_index / mesh_c;
    const double position_y = lattice_vectors[1] * a_index / mesh_a + lattice_vectors[4] * b_index / mesh_b +
                              lattice_vectors[7] * c_index / mesh_c;
    const double position_z = lattice_vectors[2] * a_index / mesh_a + lattice_vectors[5] * b_index / mesh_b +
                              lattice_vectors[8] * c_index / mesh_c;

    constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
    constexpr int n_i_spherical_functions = 2 * i_angular + 1;
    constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
    constexpr int n_j_spherical_functions = 2 * j_angular + 1;

    double i_cartesian[n_i_cartesian_functions];
    double j_cartesian[n_j_cartesian_functions];
    double i_spherical[n_i_spherical_functions];
    double j_spherical[n_j_spherical_functions];

    double density_value[n_channels];
    double prefactor[n_channels * n_i_spherical_functions * n_j_spherical_functions];

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
        density_value[i_channel] = 0;
    }

    const uint local_grid_index = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    for (int i_pair_index = accumulated_n_pairs_per_local_grid[local_grid_index];
         i_pair_index < accumulated_n_pairs_per_local_grid[local_grid_index + 1]; i_pair_index++) {

        const int i_pair = sorted_pairs_per_local_grid[i_pair_index];
        const int image_index = image_indices[i_pair];

        const double cutoff = cutoffs[i_pair];
        const double image_x = vectors_to_neighboring_images[image_index * 3];
        const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
        const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

        const int shell_pair_index = non_trivial_pairs[i_pair];
        const int j_shell_index = shell_pair_index / n_i_shells;
        const int i_shell_index = shell_pair_index % n_i_shells;
        const int i_shell = i_shells[i_shell_index];
        const int i_function = shell_to_ao_indices[i_shell];
        const int j_shell = j_shells[j_shell_index];
        const int j_function = shell_to_ao_indices[j_shell];

        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset] - image_x;
        const double i_y = env[i_coord_offset + 1] - image_y;
        const double i_z = env[i_coord_offset + 2] - image_z;
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];

        const double j_exponent = env[bas(PTR_EXP, j_shell)];
        const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
        const double j_x = env[j_coord_offset];
        const double j_y = env[j_coord_offset + 1];
        const double j_z = env[j_coord_offset + 2];
        const double j_coeff = env[bas(PTR_COEFF, j_shell)];

        const double ij_exponent = i_exponent + j_exponent;
        const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * distance_squared(
                i_x - j_x, i_y - j_y, i_z - j_z);

        const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff;
#pragma unroll
        for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            for (int i_function_index = 0; i_function_index < n_i_spherical_functions; i_function_index++) {
                for (int j_function_index = 0; j_function_index < n_j_spherical_functions; j_function_index++) {
                    const double density_matrix_value =
                            density_matrices[
                                    density_matrix_channel_stride * i_channel + image_index * density_matrix_stride +
                                    (i_function + i_function_index) * n_j_functions + j_function + j_function_index];

                    prefactor[i_channel * n_i_spherical_functions * n_j_spherical_functions +
                              i_function_index * n_j_spherical_functions + j_function_index] =
                            pair_prefactor * density_matrix_value;
                }
            }
        }

        const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
        const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
        const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

        const int lower_a_index = floor((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[0] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[1] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[2]);
        const int upper_a_index = ceil((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[0] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[1] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[2]);

        const int lower_b_index = ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[3] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[4] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[5]);
        const int upper_b_index = floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[3] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[4] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[5]);

        const int lower_c_index = ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[6] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[7] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[8]);
        const int upper_c_index = floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[6] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[7] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[8]);


        for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
            for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
                for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {

                    const double x = position_x - a_cell * lattice_vectors[0] - b_cell * lattice_vectors[3] -
                                     c_cell * lattice_vectors[6];
                    const double y = position_y - a_cell * lattice_vectors[1] - b_cell * lattice_vectors[4] -
                                     c_cell * lattice_vectors[7];
                    const double z = position_z - a_cell * lattice_vectors[2] - b_cell * lattice_vectors[5] -
                                     c_cell * lattice_vectors[8];
                    _cart_gto<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
                    _memset_sph<i_angular>(i_spherical, 1, 0);
                    _cart2sph<i_angular>(i_cartesian, i_spherical, 1, 0);

                    _cart_gto<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);
                    _memset_sph<j_angular>(j_spherical, 1, 0);
                    _cart2sph<j_angular>(j_cartesian, j_spherical, 1, 0);

                    const double r_squared = distance_squared(x - pair_x, y - pair_y, z - pair_z);
                    const double gaussian = exp(-ij_exponent * r_squared);

#pragma unroll
                    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
                        for (int i_function_index = 0; i_function_index < n_i_spherical_functions; i_function_index++) {
                            for (int j_function_index = 0;
                                 j_function_index < n_j_spherical_functions; j_function_index++) {
                                density_value[i_channel] +=
                                        prefactor[i_channel * n_i_spherical_functions * n_j_spherical_functions +
                                                  i_function_index * n_j_spherical_functions + j_function_index] *
                                        gaussian * i_spherical[i_function_index] * j_spherical[j_function_index];
                            }
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
        atomicAdd(
                density + i_channel * mesh_a * mesh_b * mesh_c + a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
                density_value[i_channel]);
    }
}


template<int n_channels>
void evaluate_density_driver(
        double *density, const double *density_matrices,
        const int left_angular, const int right_angular, const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int *mesh, const int *atm, const int *bas, const double *env, const int blocking_sizes[3]) {
    dim3 block_size(blocking_sizes[2], blocking_sizes[1], blocking_sizes[0]);
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    dim3 block_grid((mesh_c + blocking_sizes[2] - 1) / blocking_sizes[2],
                    (mesh_b + blocking_sizes[1] - 1) / blocking_sizes[1],
                    (mesh_a + blocking_sizes[0] - 1) / blocking_sizes[0]);
    if (left_angular == 0 && right_angular == 0) {
        evaluate_density_kernel<n_channels, 0, 0><<<block_grid, block_size>>>(
                density, density_matrices, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 0 && right_angular == 1) {
        evaluate_density_kernel<n_channels, 0, 1><<<block_grid, block_size>>>(
                density, density_matrices, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 1 && right_angular == 0) {
        evaluate_density_kernel<n_channels, 1, 0><<<block_grid, block_size>>>(
                density, density_matrices, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 1 && right_angular == 1) {
        evaluate_density_kernel<n_channels, 1, 1><<<block_grid, block_size>>>(
                density, density_matrices, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else {
        fprintf(stderr,
                "angular momentum pair %d, %d is not supported in evaluate_density_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());

}

template<int n_channels, int i_angular, int j_angular>
__global__ void evaluate_xc_kernel(
        double *fock, const double *xc_weights, const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int mesh_a, const int mesh_b, const int mesh_c,
        const int *atm, const int *bas, const double *env) {
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;

    if (a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c) {
        return;
    }

    const uint flattened_index = a_index * mesh_b * mesh_c + b_index * mesh_c + c_index;
    const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
    const int fock_stride = n_i_functions * n_j_functions;

    const double position_x = lattice_vectors[0] * a_index / mesh_a + lattice_vectors[3] * b_index / mesh_b +
                              lattice_vectors[6] * c_index / mesh_c;
    const double position_y = lattice_vectors[1] * a_index / mesh_a + lattice_vectors[4] * b_index / mesh_b +
                              lattice_vectors[7] * c_index / mesh_c;
    const double position_z = lattice_vectors[2] * a_index / mesh_a + lattice_vectors[5] * b_index / mesh_b +
                              lattice_vectors[8] * c_index / mesh_c;

    constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
    constexpr int n_i_spherical_functions = 2 * i_angular + 1;
    constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
    constexpr int n_j_spherical_functions = 2 * j_angular + 1;

    double xc_values[n_channels];
    double neighboring_gaussian_sum[n_i_spherical_functions * n_j_spherical_functions];

    double i_cartesian[n_i_cartesian_functions];
    double j_cartesian[n_j_cartesian_functions];
    double i_spherical[n_i_spherical_functions];
    double j_spherical[n_j_spherical_functions];

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
        xc_values[i_channel] = xc_weights[i_channel * xc_weights_stride + flattened_index];
    }

    const uint local_grid_index = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    for (int i_pair_index = accumulated_n_pairs_per_local_grid[local_grid_index];
         i_pair_index < accumulated_n_pairs_per_local_grid[local_grid_index + 1]; i_pair_index++) {

        const int i_pair = sorted_pairs_per_local_grid[i_pair_index];
        const int image_index = image_indices[i_pair];

        const double cutoff = cutoffs[i_pair];

        const double image_x = vectors_to_neighboring_images[image_index * 3];
        const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
        const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

        const int shell_pair_index = non_trivial_pairs[i_pair];
        const int j_shell_index = shell_pair_index / n_i_shells;
        const int i_shell_index = shell_pair_index % n_i_shells;
        const int i_shell = i_shells[i_shell_index];
        const int i_function = shell_to_ao_indices[i_shell];
        const int j_shell = j_shells[j_shell_index];
        const int j_function = shell_to_ao_indices[j_shell];

        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset] - image_x;
        const double i_y = env[i_coord_offset + 1] - image_y;
        const double i_z = env[i_coord_offset + 2] - image_z;
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];

        const double j_exponent = env[bas(PTR_EXP, j_shell)];
        const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
        const double j_x = env[j_coord_offset];
        const double j_y = env[j_coord_offset + 1];
        const double j_z = env[j_coord_offset + 2];
        const double j_coeff = env[bas(PTR_COEFF, j_shell)];

        const double ij_exponent = i_exponent + j_exponent;
        const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * distance_squared(
                i_x - j_x, i_y - j_y, i_z - j_z);

        const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff;

        const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
        const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
        const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

        const int lower_a_index = ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[0] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[1] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[2]);
        const int upper_a_index = floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[0] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[1] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[2]);

        const int lower_b_index = ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[3] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[4] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[5]);
        const int upper_b_index = floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[3] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[4] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[5]);

        const int lower_c_index = ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[6] +
                                       (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[7] +
                                       (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[8]);
        const int upper_c_index = floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[6] +
                                        (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[7] +
                                        (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[8]);

#pragma unroll
        for (int i_function_index = 0; i_function_index < n_i_spherical_functions; i_function_index++) {
            for (int j_function_index = 0;
                 j_function_index < n_j_spherical_functions; j_function_index++) {
                neighboring_gaussian_sum[i_function_index * n_j_spherical_functions + j_function_index] = 0;
            }
        }
        for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
            for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
                for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {

                    const double x = position_x - a_cell * lattice_vectors[0] - b_cell * lattice_vectors[3] -
                                     c_cell * lattice_vectors[6];
                    const double y = position_y - a_cell * lattice_vectors[1] - b_cell * lattice_vectors[4] -
                                     c_cell * lattice_vectors[7];
                    const double z = position_z - a_cell * lattice_vectors[2] - b_cell * lattice_vectors[5] -
                                     c_cell * lattice_vectors[8];
                    _cart_gto<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
                    _memset_sph<i_angular>(i_spherical, 1, 0);
                    _cart2sph<i_angular>(i_cartesian, i_spherical, 1, 0);

                    _cart_gto<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);
                    _memset_sph<j_angular>(j_spherical, 1, 0);
                    _cart2sph<j_angular>(j_cartesian, j_spherical, 1, 0);
                    const double r_squared = distance_squared(x - pair_x, y - pair_y, z - pair_z);
                    const double gaussian = exp(-ij_exponent * r_squared);
#pragma unroll
                    for (int i_function_index = 0; i_function_index < n_i_spherical_functions; i_function_index++) {
                        for (int j_function_index = 0;
                             j_function_index < n_j_spherical_functions; j_function_index++) {
                            neighboring_gaussian_sum[i_function_index * n_j_spherical_functions + j_function_index] +=
                                    gaussian * i_spherical[i_function_index] * j_spherical[j_function_index];
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            for (int i_function_index = 0; i_function_index < n_i_spherical_functions; i_function_index++) {
                for (int j_function_index = 0; j_function_index < n_j_spherical_functions; j_function_index++) {
                    atomicAdd(fock + fock_stride * image_index + n_j_functions * (i_function_index + i_function) +
                              j_function_index + j_function,
                              xc_values[i_channel] * pair_prefactor *
                              neighboring_gaussian_sum[i_function_index * n_j_spherical_functions + j_function_index]);
                }
            }

        }
    }
}

template<int n_channels>
void evaluate_xc_driver(
        double *fock, const double *xc_weights,
        const int left_angular, const int right_angular, const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int *mesh, const int *atm, const int *bas, const double *env, const int blocking_sizes[3]) {
    dim3 block_size(blocking_sizes[2], blocking_sizes[1], blocking_sizes[0]);
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    dim3 block_grid((mesh_c + blocking_sizes[2] - 1) / blocking_sizes[2],
                    (mesh_b + blocking_sizes[1] - 1) / blocking_sizes[1],
                    (mesh_a + blocking_sizes[0] - 1) / blocking_sizes[0]);
    if (left_angular == 0 && right_angular == 0) {
        evaluate_xc_kernel < n_channels, 0, 0 ><<<block_grid, block_size>>>(
                fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                        shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                        accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                        lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 1 && right_angular == 0) {
        evaluate_xc_kernel < n_channels, 1, 0 ><<<block_grid, block_size>>>(
                fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                        shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                        accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                        lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 0 && right_angular == 1) {
        evaluate_xc_kernel < n_channels, 0, 1 ><<<block_grid, block_size>>>(
                fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                        shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                        accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                        lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else if (left_angular == 1 && right_angular == 1) {
        evaluate_xc_kernel < n_channels, 1, 1 ><<<block_grid, block_size>>>(
                fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, n_i_shells, j_shells,
                shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
                accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    } else {
        fprintf(stderr,
                "angular momentum pair %d, %d is not supported in evaluate_xc_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

extern "C" {
void evaluate_density_driver(
        double *density, const double *density_matrices,
        const int left_angular, const int right_angular, const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int *mesh, const int *atm, const int *bas, const double *env, const int blocking_sizes[3],
        const int n_channels) {
    if (n_channels == 1) {
        evaluate_density_driver<1>(
                density, density_matrices, left_angular, right_angular, non_trivial_pairs, cutoffs,
                i_shells, n_i_shells, j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
                sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
                image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    } else if (n_channels == 2) {
        evaluate_density_driver<2>(
                density, density_matrices, left_angular, right_angular, non_trivial_pairs, cutoffs,
                i_shells, n_i_shells, j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
                sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
                image_indices, vectors_to_neighboring_images, n_images,
                lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    } else {
        fprintf(stderr, "n_channels more than 2 is not supported.\n");
    }
}

void evaluate_xc_driver(
        double *fock, const double *xc_weights, const int left_angular, const int right_angular,
        const int *non_trivial_pairs, const double *cutoffs,
        const int *i_shells, const int n_i_shells, const int *j_shells,
        const int *shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
        const int *sorted_pairs_per_local_grid, const int *accumulated_n_pairs_per_local_grid,
        const int *image_indices, const double *vectors_to_neighboring_images, const int n_images,
        const double *lattice_vectors, const double *reciprocal_lattice_vectors,
        const int *mesh, const int *atm, const int *bas, const double *env, const int blocking_sizes[3],
        const int n_channels) {
    if (n_channels == 1) {
        evaluate_xc_driver<1>(fock, xc_weights, left_angular, right_angular, non_trivial_pairs, cutoffs,
                              i_shells, n_i_shells, j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
                              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
                              image_indices, vectors_to_neighboring_images, n_images,
                              lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    } else if (n_channels == 2) {
        evaluate_xc_driver<2>(fock, xc_weights, left_angular, right_angular, non_trivial_pairs, cutoffs,
                              i_shells, n_i_shells, j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,
                              sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,
                              image_indices, vectors_to_neighboring_images, n_images,
                              lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    } else {
        fprintf(stderr, "n_channels more than 2 is not supported.\n");
    }
}
}
