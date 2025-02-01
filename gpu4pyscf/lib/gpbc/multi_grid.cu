#include <cstdio>
#include <cuda_runtime.h>
#include <gint/config.h>
#include <gint/gint.h>
#include <cassert>
#include <ios>
#include <gint/cuda_alloc.cuh>

#define atm(SLOT, I)     atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I)     bas[BAS_SLOTS * (I) + (SLOT)]

#define EIJCUTOFF       60
#define PTR_EXPDROP     16

__host__ __device__ double distance_squared(const double x, const double y, const double z) {
    return x * x + y * y + z * z;
}

__global__ void evaluate_xc_kernel_ss_orthogonal(
        double *fock, const double *xc_weights, const int i_shell_begin, const int i_shell_end,
        const int j_shell_begin, const int j_shell_end,
        const int *shell_to_function,
        const int n_images, const double *vectors_to_neighboring_images,
        const double a, const double b, const double c,
        const int offset_a, const int offset_b, const int offset_c,
        const int local_mesh_a, const int local_mesh_b, const int local_mesh_c,
        const int global_mesh_a, const int global_mesh_b, const int global_mesh_c,
        const int *atm, const int *bas, const double *env) {
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
    const uint flattened_index = a_index * local_mesh_b * local_mesh_c + b_index * local_mesh_c + c_index;

    if (a_index >= local_mesh_a || b_index >= local_mesh_b || c_index >= local_mesh_c) {
        return;
    }

    const double position_x = a * (a_index + offset_a) / global_mesh_a;
    const double position_y = b * (b_index + offset_b) / global_mesh_b;
    const double position_z = c * (c_index + offset_c) / global_mesh_c;
    const int i_function_begin = shell_to_function[i_shell_begin];
    const int j_function_begin = shell_to_function[j_shell_begin];
    const int n_ao_i = shell_to_function[i_shell_end] - i_function_begin;
    const int n_ao_j = shell_to_function[j_shell_end] - j_function_begin;
    const int fock_stride = n_ao_i * n_ao_j;

    const double xc_value = xc_weights[flattened_index];

    for (int i_shell = i_shell_begin; i_shell < i_shell_end; i_shell++) {
        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset];
        const double i_y = env[i_coord_offset + 1];
        const double i_z = env[i_coord_offset + 2];
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];
        const int i_function_index = shell_to_function[i_shell] - i_function_begin;

        for (int j_shell = j_shell_begin; j_shell < j_shell_end; j_shell++) {
            const double j_exponent = env[bas(PTR_EXP, j_shell)];
            const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
            const double j_x = env[j_coord_offset];
            const double j_y = env[j_coord_offset + 1];
            const double j_z = env[j_coord_offset + 2];
            const double j_coeff = env[bas(PTR_COEFF, j_shell)];
            const int j_function_index = shell_to_function[j_shell] - j_function_begin;

            const double ij_exponent = i_exponent + j_exponent;
            for (int i_image = 0; i_image < n_images; i_image++) {
                const double i_image_x = vectors_to_neighboring_images[i_image * 3];
                const double i_image_y = vectors_to_neighboring_images[i_image * 3 + 1];
                const double i_image_z = vectors_to_neighboring_images[i_image * 3 + 2];

                const double shifted_i_x = i_x - i_image_x;
                const double shifted_i_y = i_y - i_image_y;
                const double shifted_i_z = i_z - i_image_z;

                const double ij_norm_squared = distance_squared(shifted_i_x - j_x,
                                                                shifted_i_y - j_y,
                                                                shifted_i_z - j_z);

                const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * ij_norm_squared;

                if (ij_exponent_in_prefactor > EIJCUTOFF) {
                    continue;
                }
                const double prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff * 0.282094791773878143 *
                                         0.282094791773878143 * xc_value;

                const double x = position_x - (i_exponent * shifted_i_x + j_exponent * j_x) / ij_exponent;
                const double y = position_y - (i_exponent * shifted_i_y + j_exponent * j_y) / ij_exponent;
                const double z = position_z - (i_exponent * shifted_i_z + j_exponent * j_z) / ij_exponent;

                const double real_space_cutoff = sqrt(EIJCUTOFF / ij_exponent);

                const int lower_a_index = ceil((x - real_space_cutoff) / a);
                const int upper_a_index = floor((x + real_space_cutoff) / a);
                const int lower_b_index = ceil((y - real_space_cutoff) / b);
                const int upper_b_index = floor((y + real_space_cutoff) / b);
                const int lower_c_index = ceil((z - real_space_cutoff) / c);
                const int upper_c_index = floor((z + real_space_cutoff) / c);

                if (upper_a_index >= lower_a_index && upper_b_index >= lower_b_index &&
                    upper_c_index >= lower_c_index) {
                    double neighboring_gaussian_sum = 0;

                    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
                        for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
                            for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {
                                const double r_squared = distance_squared(
                                        x - a_cell * a, y - b_cell * b, z - c_cell * c);

                                neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                            }
                        }
                    }

                    atomicAdd(fock + fock_stride * i_image + n_ao_i * j_function_index + i_function_index,
                              prefactor * neighboring_gaussian_sum);
                }


            }

            // TODO: def possible to reduce within a block first

        }
    }
}

__global__ void evaluate_density_kernel_ss_orthogonal(
        double *density, const double *density_matrices,
        const int i_shell_begin, const int i_shell_end,
        const int j_shell_begin, const int j_shell_end,
        const int *shell_to_function,
        const int n_images, const double *vectors_to_neighboring_images,
        const double a, const double b, const double c,
        const int offset_a, const int offset_b, const int offset_c,
        const int local_mesh_a, const int local_mesh_b, const int local_mesh_c,
        const int global_mesh_a, const int global_mesh_b, const int global_mesh_c,
        const int *atm, const int *bas, const double *env) {
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;

    if (a_index >= local_mesh_a || b_index >= local_mesh_b || c_index >= local_mesh_c) {
        return;
    }

    const double position_x = a * (a_index + offset_a) / global_mesh_a;
    const double position_y = b * (b_index + offset_b) / global_mesh_b;
    const double position_z = c * (c_index + offset_c) / global_mesh_c;
    const int i_function_begin = shell_to_function[i_shell_begin];
    const int j_function_begin = shell_to_function[j_shell_begin];
    const int n_ao_i = shell_to_function[i_shell_end] - i_function_begin;
    const int n_ao_j = shell_to_function[j_shell_end] - j_function_begin;
    const int density_matrix_stride = n_ao_i * n_ao_j;
    double density_value = 0;

    for (int i_shell = i_shell_begin; i_shell < i_shell_end; i_shell++) {
        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset];
        const double i_y = env[i_coord_offset + 1];
        const double i_z = env[i_coord_offset + 2];
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];
        const int i_function_index = shell_to_function[i_shell] - i_function_begin;

        for (int j_shell = j_shell_begin; j_shell < j_shell_end; j_shell++) {
            const double j_exponent = env[bas(PTR_EXP, j_shell)];
            const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
            const double j_x = env[j_coord_offset];
            const double j_y = env[j_coord_offset + 1];
            const double j_z = env[j_coord_offset + 2];
            const double j_coeff = env[bas(PTR_COEFF, j_shell)];
            const int j_function_index = shell_to_function[j_shell] - j_function_begin;

            const double ij_exponent = i_exponent + j_exponent;

            for (int i_image = 0; i_image < n_images; i_image++) {
                const double i_image_x = vectors_to_neighboring_images[i_image * 3];
                const double i_image_y = vectors_to_neighboring_images[i_image * 3 + 1];
                const double i_image_z = vectors_to_neighboring_images[i_image * 3 + 2];

                const double shifted_i_x = i_x - i_image_x;
                const double shifted_i_y = i_y - i_image_y;
                const double shifted_i_z = i_z - i_image_z;

                const double ij_norm_squared = distance_squared(shifted_i_x - j_x,
                                                                shifted_i_y - j_y,
                                                                shifted_i_z - j_z);

                const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * ij_norm_squared;

                if (ij_exponent_in_prefactor > EIJCUTOFF) {
                    continue;
                }

                const double prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff
                                         * 0.282094791773878143 * 0.282094791773878143 *
                                         density_matrices[i_image * density_matrix_stride + j_function_index * n_ao_i +
                                                          i_function_index];

                const double x = position_x - (i_exponent * shifted_i_x + j_exponent * j_x) / ij_exponent;
                const double y = position_y - (i_exponent * shifted_i_y + j_exponent * j_y) / ij_exponent;
                const double z = position_z - (i_exponent * shifted_i_z + j_exponent * j_z) / ij_exponent;

                const double real_space_cutoff = sqrt(EIJCUTOFF / ij_exponent);

                const int lower_a_index = ceil((x - real_space_cutoff) / a);
                const int upper_a_index = floor((x + real_space_cutoff) / a);
                const int lower_b_index = ceil((y - real_space_cutoff) / b);
                const int upper_b_index = floor((y + real_space_cutoff) / b);
                const int lower_c_index = ceil((z - real_space_cutoff) / c);
                const int upper_c_index = floor((z + real_space_cutoff) / c);


                if (upper_a_index >= lower_a_index && upper_b_index >= lower_b_index &&
                    upper_c_index >= lower_c_index) {
                    double neighboring_gaussian_sum = 0;

                    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
                        for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
                            for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {
                                const double r_squared = distance_squared(
                                        x - a_cell * a, y - b_cell * b, z - c_cell * c);

                                neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                            }
                        }
                    }

                    density_value += prefactor * neighboring_gaussian_sum;
                }
            }
        }
    }
    density[a_index * local_mesh_b * local_mesh_c + b_index * local_mesh_c + c_index] = density_value;
}

extern "C" {
void evaluate_xc_driver(
        double *fock, const double *xc_weights, const int *shells_slice, const int *shell_to_function,
        const int left_angular, const int right_angular,
        const int n_images, const double *vectors_to_neighboring_images,
        const double *lattice_vector, const int *offset, const int *local_mesh, const int *global_mesh,
        const int *atm, const int *bas, const double *env) {
    dim3 block_size(4, 4, 4);
    int local_mesh_a = local_mesh[0];
    int local_mesh_b = local_mesh[1];
    int local_mesh_c = local_mesh[2];
    dim3 block_grid((local_mesh_c + 3) / 4, (local_mesh_b + 3) / 4, (local_mesh_a + 3) / 4);

    const double a = lattice_vector[0];
    const double b = lattice_vector[4];
    const double c = lattice_vector[8];

    if (left_angular == 0 && right_angular == 0) {
        evaluate_xc_kernel_ss_orthogonal<<<block_grid, block_size>>>(fock, xc_weights, shells_slice[0],
                                                                     shells_slice[1], shells_slice[2],
                                                                     shells_slice[3],
                                                                     shell_to_function,
                                                                     n_images, vectors_to_neighboring_images,
                                                                     a, b, c,
                                                                     offset[0], offset[1], offset[2],
                                                                     local_mesh_a, local_mesh_b, local_mesh_c,
                                                                     global_mesh[0], global_mesh[1], global_mesh[2],
                                                                     atm, bas, env);
    } else {
        fprintf(stderr, "angular momentum pair %d, %d is not supported in evaluate_xc_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

void evaluate_density_driver(
        double *density, const double *density_matrices, const int *shells_slice, const int *shell_to_function,
        const int left_angular, const int right_angular,
        const int n_images, const double *vectors_to_neighboring_images,
        const double *lattice_vector, const int *offset, const int *local_mesh, const int *global_mesh,
        const int *atm, const int *bas, const double *env) {
    dim3 block_size(4, 4, 4);
    int local_mesh_a = local_mesh[0];
    int local_mesh_b = local_mesh[1];
    int local_mesh_c = local_mesh[2];
    dim3 block_grid((local_mesh_c + 3) / 4, (local_mesh_b + 3) / 4, (local_mesh_a + 3) / 4);

    const double a = lattice_vector[0];
    const double b = lattice_vector[4];
    const double c = lattice_vector[8];

    if (left_angular == 0 && right_angular == 0) {
        evaluate_density_kernel_ss_orthogonal<<<block_grid, block_size>>>(density, density_matrices, shells_slice[0],
                                                                          shells_slice[1], shells_slice[2],
                                                                          shells_slice[3],
                                                                          shell_to_function,
                                                                          n_images, vectors_to_neighboring_images,
                                                                          a, b, c,
                                                                          offset[0], offset[1], offset[2],
                                                                          local_mesh_a, local_mesh_b, local_mesh_c,
                                                                          global_mesh[0], global_mesh[1], global_mesh[2],
                                                                          atm, bas, env);
    } else {
        fprintf(stderr, "angular momentum pair %d, %d is not supported in evaluate_density_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
    // checkCudaErrors(cudaDeviceSynchronize());
}
}

//
//
// static const int n_functions_for_l[] = {
//     1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
// };
//
// static const int cumulative_n_functions[] = {
//     1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816,
// };
//
// static int max_recursive_relation_size[] = {
//     1, 4, 12, 30, 60, 120, 210, 350, 560, 840, 1260, 1800, 2520, 3465, 4620,
//     6160, 8008, 10296, 13104, 16380, 20475,
// };
//
// static int _MAX_AFFINE_SIZE[] = {
//     1, 8, 32, 108, 270, 640, 1280, 2500, 4375, 7560, 12096, 19208, 28812,
//     43008, 61440, 87480,
// };
//
// static int _has_overlap(int nx0, int nx1, int nx_per_cell)
// {
//     return nx0 < nx1 + 3;
// }
//
// static int _num_grids_on_x(int nimgx, int nx0, int nx1, int nx_per_cell)
// {
//     int ngridx;
//     if (nimgx == 1) {
//         ngridx = nx1 - nx0;
//     } else if (nimgx == 2 && !_has_overlap(nx0, nx1, nx_per_cell)) {
//         ngridx = nx1 - nx0 + nx_per_cell;
//     } else {
//         ngridx = nx_per_cell;
//     }
//     return ngridx;
// }
//
// static double gto_rcut(double alpha, int l, double c, double log_prec)
// {
//     double log_c = log(fabs(c));
//     double prod = 0;
//     double r = 10.;
//     double log_2a = log(2*alpha);
//     double log_r = log(r);
//
//     if (2*log_r + log_2a > 1) { // r^2 >~ 3/(2a)
//         prod = (l+1) * log_r - log_2a;
//     } else {
//         prod = -(l+4)/2 * log_2a;
//     }
//
//     //log_r = .5 * (prod / alpha);
//     //if (2*log_r + log_2a > 1) {
//     //        prod = (l+1) * log_r - log_2a;
//     //} else {
//     //        prod = -(l+4)/2 * log_2a;
//     //}
//
//     prod += log_c - log_prec;
//     if (prod < alpha) {
//         // if rcut < 1, estimating based on exp^{-a*rcut^2}
//         prod = log_c - log_prec;
//     }
//     if (prod > 0) {
//         r = sqrt(prod / alpha);
//     } else {
//         r = 0;
//     }
//     return r;
// }
//
//
// static int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
//                             double a, double b, double cutoff,
//                             double xi, double xj, double ai, double aj,
//                             int periodic, int nx_per_cell, int topl,
//                             int offset, int submesh, double *cache)
// {
//         double aij = ai + aj;
//         double xij = (ai * xi + aj * xj) / aij;
//         double heights_inv = b;
//         double xij_frac = xij * heights_inv;
//         double edge0 = xij_frac - cutoff * heights_inv;
//         double edge1 = xij_frac + cutoff * heights_inv;
//         if (edge0 == edge1) {
// // cutoff may be so small that it does not provide difference to edge0 and
// // edge1. When edge0 and edge1 are right on the edge of the box (== integer),
// // nimg0 may be equal to nimg1 and nimg can be 0.  Skip this extreme condition.
//                 return 0;
//         }
//
//         int nimg0 = 0;
//         int nimg1 = 1;
// // If submesh is not identical to mesh, it means the product of the basis
// // functions should be completely inside the unit cell. Only one image needs to
// // be considered.
//         if (offset != 0 || submesh != nx_per_cell) {
// // |i> is the steep function and centered inside image 0. Moving |j> all around
// // will not change the center of |ij>. The periodic system can be treated as
// // non-periodic system so that only one image needs to be considered.
//                 nimg0 = (int)floor(xij_frac);
//                 nimg1 = nimg0 + 1;
//                 edge0 = MAX(edge0, nimg0);
//                 edge1 = MIN(edge1, nimg1);
//         } else if (periodic) {
//                 nimg0 = (int)floor(edge0);
//                 nimg1 = (int)ceil (edge1);
//         }
//         int nimg = nimg1 - nimg0;
//         int nmx0 = nimg0 * nx_per_cell;
//         int nmx1 = nimg1 * nx_per_cell;
//         int nmx = nmx1 - nmx0;
//
//         int nx0 = (int)floor(edge0 * nx_per_cell);
//         int nx1 = (int)ceil (edge1 * nx_per_cell);
//         int nx0_edge;
//         int nx1_edge;
//         // to ensure nx0, nx1 being inside the unit cell
//         if (periodic) {
//                 nx0 = (nx0 - nmx0) % nx_per_cell;
//                 nx1 = (nx1 - nmx0) % nx_per_cell;
//                 if (nx1 == 0) {
//                         nx1 = nx_per_cell;
//                 }
//         }
//         // If only 1 image is required, after drawing the grids to the unit cell
//         // as above, the periodic system can be treated as a non-periodic
//         // system, which requires [nx0:nx1] being inside submesh.  It is
//         // necessary because xij+/-cutoff may be out of the submesh for periodic
//         // systems when offset and submesh are specified.
//         if (nimg == 1) {
//                 nx0 = MIN(nx0, offset + submesh);
//                 nx0 = MAX(nx0, offset);
//                 nx1 = MIN(nx1, offset + submesh);
//                 nx1 = MAX(nx1, offset);
//                 nx0_edge = nx0;
//                 nx1_edge = nx1;
//         } else {
//                 nx0_edge = 0;
//                 nx1_edge = nmx;
//         }
//         img_slice[0] = nimg0;
//         img_slice[1] = nimg1;
//         grid_slice[0] = nx0;
//         grid_slice[1] = nx1;
//
//         int ngridx = _num_grids_on_x(nimg, nx0, nx1, nx_per_cell);
//         if (ngridx == 0) {
//                 return 0;
//         }
//
//         int i, m, l;
//         double *px0;
//
//         double *gridx = cache;
//         double *xs_all = cache + nmx;
//         if (nimg == 1) {
//                 xs_all = xs_exp;
//         }
//         int grid_close_to_xij = rint(xij_frac * nx_per_cell) - nmx0;
//         grid_close_to_xij = MIN(grid_close_to_xij, nx1_edge);
//         grid_close_to_xij = MAX(grid_close_to_xij, nx0_edge);
//
//         double img0_x = a * nimg0;
//         double dx = a / nx_per_cell;
//         double base_x = img0_x + dx * grid_close_to_xij;
//         double x0xij = base_x - xij;
//         double _x0x0 = -aij * x0xij * x0xij;
//         if (_x0x0 < EXPMIN) {
//                 return 0;
//         }
//
//         double _dxdx = -aij * dx * dx;
//         double _x0dx = -2 * aij * x0xij * dx;
//         double exp_dxdx = exp(_dxdx);
//         double exp_2dxdx = exp_dxdx * exp_dxdx;
//         double exp_x0dx = exp(_x0dx + _dxdx);
//         double exp_x0x0 = exp(_x0x0);
//
//         for (i = grid_close_to_xij; i < nx1_edge; i++) {
//                 xs_all[i] = exp_x0x0;
//                 exp_x0x0 *= exp_x0dx;
//                 exp_x0dx *= exp_2dxdx;
//         }
//
//         exp_x0dx = exp(_dxdx - _x0dx);
//         exp_x0x0 = exp(_x0x0);
//         for (i = grid_close_to_xij-1; i >= nx0_edge; i--) {
//                 exp_x0x0 *= exp_x0dx;
//                 exp_x0dx *= exp_2dxdx;
//                 xs_all[i] = exp_x0x0;
//         }
//
//         if (topl > 0) {
//                 double x0xi = img0_x - xi;
//                 for (i = nx0_edge; i < nx1_edge; i++) {
//                         gridx[i] = x0xi + i * dx;
//                 }
//                 for (l = 1; l <= topl; l++) {
//                         px0 = xs_all + (l-1) * nmx;
//                         for (i = nx0_edge; i < nx1_edge; i++) {
//                                 px0[nmx+i] = px0[i] * gridx[i];
//                         }
//                 }
//         }
//
//         if (nimg > 1) {
//                 for (l = 0; l <= topl; l++) {
//                         px0 = xs_all + l * nmx;
//                         for (i = 0; i < nx_per_cell; i++) {
//                                 xs_exp[l*nx_per_cell+i] = px0[i];
//                         }
//                         for (m = 1; m < nimg; m++) {
//                                 px0 = xs_all + l * nmx + m*nx_per_cell;
//                                 for (i = 0; i < nx_per_cell; i++) {
//                                         xs_exp[l*nx_per_cell+i] += px0[i];
//                                 }
//                         }
//                 }
//         }
//         return ngridx;
// }
//
// static int _init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
//                            int *img_slice, int *grid_slice,
//                            int *offset, int *submesh, int *mesh,
//                            int topl, int dimension, double cutoff,
//                            double ai, double aj, double *ri, double *rj,
//                            double *a, double *b, double *cache)
// {
//     int l1 = topl + 1;
//     *xs_exp = cache;
//     *ys_exp = *xs_exp + l1 * mesh[0];
//     *zs_exp = *ys_exp + l1 * mesh[1];
//     int data_size = l1 * (mesh[0] + mesh[1] + mesh[2]);
//     cache += data_size;
//
//     int ngridx = _orth_components(*xs_exp, img_slice, grid_slice,
//                                   a[0], b[0], cutoff, ri[0], rj[0], ai, aj,
//                                   (dimension>=1), mesh[0], topl,
//                                   offset[0], submesh[0], cache);
//     if (ngridx == 0) {
//         return 0;
//     }
//
//     int ngridy = _orth_components(*ys_exp, img_slice+2, grid_slice+2,
//                                   a[4], b[4], cutoff, ri[1], rj[1], ai, aj,
//                                   (dimension>=2), mesh[1], topl,
//                                   offset[1], submesh[1], cache);
//     if (ngridy == 0) {
//         return 0;
//     }
//
//     int ngridz = _orth_components(*zs_exp, img_slice+4, grid_slice+4,
//                                   a[8], b[8], cutoff, ri[2], rj[2], ai, aj,
//                                   (dimension>=3), mesh[2], topl,
//                                   offset[2], submesh[2], cache);
//     if (ngridz == 0) {
//         return 0;
//     }
//
//     return data_size;
// }
//
// static void _orth_ints(double *out, double *weights,
//                        int floorl, int topl, double fac,
//                        double *xs_exp, double *ys_exp, double *zs_exp,
//                        int *img_slice, int *grid_slice,
//                        int *offset, int *submesh, int *mesh, double *cache)
// {
//         int l1 = topl + 1;
//         int nimgx0 = img_slice[0];
//         int nimgx1 = img_slice[1];
//         int nimgy0 = img_slice[2];
//         int nimgy1 = img_slice[3];
//         int nimgz0 = img_slice[4];
//         int nimgz1 = img_slice[5];
//         int nimgx = nimgx1 - nimgx0;
//         int nimgy = nimgy1 - nimgy0;
//         int nimgz = nimgz1 - nimgz0;
//         int nx0 = grid_slice[0];
//         int nx1 = grid_slice[1];
//         int ny0 = grid_slice[2];
//         int ny1 = grid_slice[3];
//         int nz0 = grid_slice[4];
//         int nz1 = grid_slice[5];
//         int ngridx = _num_grids_on_x(nimgx, nx0, nx1, mesh[0]);
//         int ngridy = _num_grids_on_x(nimgy, ny0, ny1, mesh[1]);
//         //int ngridz = _num_grids_on_x(nimgz, nz0, nz1, mesh[2]);
//
//         const char TRANS_N = 'N';
//         const double D0 = 0;
//         const double D1 = 1;
//         int xcols = mesh[1] * mesh[2];
//         int ycols = mesh[2];
//         double *weightyz = cache;
//         double *weightz = weightyz + l1*xcols;
//         double *pz, *pweightz;
//         double val;
//         int lx, ly, lz;
//         int l, i, n;
//
//         //TODO: optimize the case in which nimgy << mesh[1] and nimgz << mesh[2]
//         if (nimgx == 1) {
//                 dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
//                        &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
//                        &D0, weightyz, &xcols);
//         } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
//                 dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
//                        &fac, weights, &xcols, xs_exp, mesh,
//                        &D0, weightyz, &xcols);
//                 ngridx = mesh[0] - nx0;
//                 dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
//                        &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
//                        &D1, weightyz, &xcols);
//         } else {
//                 dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
//                        &fac, weights, &xcols, xs_exp, mesh,
//                        &D0, weightyz, &xcols);
//         }
//
//         if (nimgy == 1) {
//                 for (lx = 0; lx <= topl; lx++) {
//                         dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
//                                &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
//                                &D0, weightz+lx*l1*ycols, &ycols);
//                         // call _orth_dot_z if ngridz << nimgz
//                 }
//         } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
//                 ngridy = mesh[1] - ny0;
//                 for (lx = 0; lx <= topl; lx++) {
//                         dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ny1,
//                                &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
//                                &D0, weightz+lx*l1*ycols, &ycols);
//                         dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
//                                &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
//                                &D1, weightz+lx*l1*ycols, &ycols);
//                         // call _orth_dot_z if ngridz << nimgz
//                 }
//         } else {
//                 for (lx = 0; lx <= topl; lx++) {
//                         dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
//                                &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
//                                &D0, weightz+lx*l1*ycols, &ycols);
//                 }
//         }
//
//         if (nimgz == 1) {
//                 for (n = 0, l = floorl; l <= topl; l++) {
//                 for (lx = l; lx >= 0; lx--) {
//                 for (ly = l - lx; ly >= 0; ly--, n++) {
//                         lz = l - lx - ly;
//                         pz = zs_exp + lz * mesh[2];
//                         pweightz = weightz + (lx * l1 + ly) * mesh[2];
//                         val = 0;
//                         for (i = nz0; i < nz1; i++) {
//                                 val += pweightz[i] * pz[i];
//                         }
//                         out[n] = val;
//                 } } }
//         } else if (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
//                 for (n = 0, l = floorl; l <= topl; l++) {
//                 for (lx = l; lx >= 0; lx--) {
//                 for (ly = l - lx; ly >= 0; ly--, n++) {
//                         lz = l - lx - ly;
//                         pz = zs_exp + lz * mesh[2];
//                         pweightz = weightz + (lx * l1 + ly) * mesh[2];
//                         val = 0;
//                         for (i = 0; i < nz1; i++) {
//                                 val += pweightz[i] * pz[i];
//                         }
//                         for (i = nz0; i < mesh[2]; i++) {
//                                 val += pweightz[i] * pz[i];
//                         }
//                         out[n] = val;
//                 } } }
//         } else {
//                 for (n = 0, l = floorl; l <= topl; l++) {
//                 for (lx = l; lx >= 0; lx--) {
//                 for (ly = l - lx; ly >= 0; ly--, n++) {
//                         lz = l - lx - ly;
//                         pz = zs_exp + lz * mesh[2];
//                         pweightz = weightz + (lx * l1 + ly) * mesh[2];
//                         val = 0;
//                         for (i = 0; i < mesh[2]; i++) {
//                                 val += pweightz[i] * pz[i];
//                         }
//                         out[n] = val;
//                 } } }
//         }
// }
//
// int NUMINTeval_lda_orth(double *weights, double *out, int n_components,
//                         int li, int lj, double ai, double aj,
//                         double *ri, double *rj, double fac, double log_prec,
//                         int dimension, double *a, double *b,
//                         int *offset, int *submesh, int *mesh, double *cache)
// {
//     int floorl = li;
//     int topl = li + lj;
//     int offset_g1d = cumulative_n_functions[floorl] - n_functions_for_l[floorl];
//     int len_g3d = cumulative_n_functions[topl] - offset_g1d;
//     double cutoff = gto_rcut(ai+aj, topl, fac, log_prec);
//     double *g3d = cache;
//     cache += len_g3d;
//     int img_slice[6];
//     int grid_slice[6];
//     double *xs_exp, *ys_exp, *zs_exp;
//     int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
//                                     grid_slice, offset, submesh, mesh,
//                                     topl, dimension, cutoff,
//                                     ai, aj, ri, rj, a, b, cache);
//     if (data_size == 0) {
//         return 0;
//     }
//     cache += data_size;
//
//     _orth_ints(g3d, weights, floorl, topl, fac, xs_exp, ys_exp, zs_exp,
//                img_slice, grid_slice, offset, submesh, mesh, cache);
//
//     cache = g3d + max_recursive_relation_size[topl];
//     _plain_vrr2d(out, g3d, cache, li, lj, ri, rj);
//     return 1;
// }
//
// static int _rho_cache_size(int l, int comp, int* mesh)
// {
//     int l1 = l * 2 + 1;
//     int cache_size = 0;
//     cache_size += l1 * mesh[1] * mesh[2];
//     cache_size += l1 * l1 * mesh[2] * 2;
//     cache_size = MAX(cache_size, 3*max_recursive_relation_size[l*2]);
//     cache_size = MAX(cache_size, cumulative_n_functions[l*2]+2*_MAX_AFFINE_SIZE[l*2]);
//     cache_size += l1 * (mesh[0] + mesh[1] + mesh[2]);
//     cache_size += l1 * l1 * l1;
//     return cache_size + 1000000;
// }
//
// static void _apply_rho(void (*eval_rho)(), double *rho, double *dm,
//                        size_t *dims, int comp,
//                        double log_prec, int dimension, double *a, double *b,
//                        int *offset, int *submesh, int *mesh, int *shls,
//                        int *atm, int natm, int *bas, int nbas, double *env,
//                        double *cache)
// {
//     const size_t naoi = dims[0];
//     const int i_sh = shls[0];
//     const int j_sh = shls[1];
//     const int li = bas(ANG_OF, i_sh);
//     const int lj = bas(ANG_OF, j_sh);
//     double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i_sh));
//     double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j_sh));
//     double ai = env[bas(PTR_EXP, i_sh)];
//     double aj = env[bas(PTR_EXP, j_sh)];
//     double ci = env[bas(PTR_COEFF, i_sh)];
//     double cj = env[bas(PTR_COEFF, j_sh)];
//     double aij = ai + aj;
//     double rrij = CINTsquare_dist(ri, rj); // definition not found
//     double eij = (ai * aj / aij) * rrij;
//     if (eij > EIJCUTOFF) {
//         return;
//     }
//     double fac = exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj); // guessing some easy factor
//     if (fac < env[PTR_EXPDROP]) { // guessing some cutoff
//         return;
//     }
//
//     (*eval_rho)(rho, dm, comp, naoi, li, lj, ai, aj, ri, rj,
//                 fac, log_prec, dimension, a, b,
//                 offset, submesh, mesh, cache);
// }
//
// void GPBC_density_driver(void (*density_kernel)(), double* density, double* density_matrices,
//                          int n_components, int hermi, int* shells_slice, int* shell_to_function,
//                          double precision_in_log, int dimension, int n_images, double* vectors_to_nearest_images,
//                          double* lattice_vector, double* reciprocal_lattice_vector, int* offset, int* lattice_sum_mesh,
//                          int* global_mesh,
//                          int* atm, int n_atoms, int* bas, int n_basis, double* env,
//                          int n_env)
// {
//     int i_shell_begin = shells_slice[0];
//     int i_shell_end = shells_slice[1];
//     int j_shell_begin = shells_slice[2];
//     int j_shell_end = shells_slice[3];
//     int n_i_shells = i_shell_end - i_shell_begin;
//     int n_j_shells = j_shell_end - j_shell_begin;
//     size_t n_ao_i = shell_to_function[i_shell_end] - shell_to_function[i_shell_begin];
//     size_t n_ao_j = shell_to_function[j_shell_end] - shell_to_function[j_shell_begin];
//     size_t n_ao_i_squared = n_ao_i * n_ao_i;
//
//     int lmax = 0;
//     int i_basis;
//     for (i_basis = 0; i_basis < n_basis; i_basis++)
//     {
//         lmax = MAX(lmax, bas(ANG_OF, i_basis));
//     }
//     int cache_size = _rho_cache_size(lmax, n_components, lattice_sum_mesh);
//     size_t n_grid_points_for_summation = ((size_t)lattice_sum_mesh[0]) * lattice_sum_mesh[1] * lattice_sum_mesh[2];
//
//     // Usually 3D cell
//     if (dimension == 0)
//     {
//         n_images = 1;
//     }
//
//     double* density_buffer[MAX_THREADS];
// #pragma omp parallel
//     {
//         size_t n_ao_pairs = n_ao_i * n_ao_j;
//         size_t n_shell_pairs = n_i_shells * n_j_shells;
//         size_t dims[] = {n_ao_i, n_ao_j};
//         size_t ij_shell_pair_in_image_m;
//         int i_shell, j_shell, ij_shell_pair, m_image, i0, j0;
//         int shell_indices[2];
//         double* cache = malloc(sizeof(double) * cache_size);
//         double* env_loc = malloc(sizeof(double) * n_env);
//         cudaMemcpy(env_loc, env, n_env * sizeof(double), cudaMemcpyHostToHost);
//         int xyz_offset_for_i_shell;
//         int thread_id = omp_get_thread_num();
//         double *rho_priv, *density_matrix_pointer;
//         if (thread_id == 0)
//         {
//             density_buffer[thread_id] = density;
//         }
//         else
//         {
//             cudaMallocHost(&density_buffer[thread_id], n_components * n_grid_points_for_summation * sizeof(double));
//         }
//         if (hermi == 1)
//         {
//             // Note hermitian character of the density matrices can only be found by
//             // rearranging the repeated images:
//             //     dmR - dmR[::-1].transpose(0,2,1) == 0
// #pragma omp for schedule(static)
//             // seems to be making the density matrix upper-diagonal. I don't think this is a necessary step though.
//             for (m_image = 0; m_image < n_images; m_image++)
//             {
//                 density_matrix_pointer = density_matrices + m_image * n_ao_i_squared;
//                 for (j0 = 1; j0 < n_ao_i; j0++)
//                 {
//                     for (i0 = 0; i0 < j0; i0++)
//                     {
//                         density_matrix_pointer[j0 * n_ao_i + i0] *= 2;
//                         density_matrix_pointer[i0 * n_ao_i + j0] = 0;
//                     }
//                 }
//             }
//         }
//
// #pragma omp for schedule(dynamic)
//         for (ij_shell_pair_in_image_m = 0; ij_shell_pair_in_image_m < n_images * n_shell_pairs; ij_shell_pair_in_image_m++)
//         {
//             m_image = ij_shell_pair_in_image_m / n_shell_pairs;
//             ij_shell_pair = ij_shell_pair_in_image_m % n_shell_pairs;
//             i_shell = ij_shell_pair / n_j_shells;
//             j_shell = ij_shell_pair % n_j_shells;
//             if (hermi != 1 && i_shell > j_shell) // Only upper triangle of the density matrix is computed
//             {
//                 continue;
//             }
//
//             i_shell += i_shell_begin;
//             j_shell += j_shell_begin;
//             shell_indices[0] = i_shell;
//             shell_indices[1] = j_shell;
//             i0 = shell_to_function[i_shell] - shell_to_function[i_shell_begin];
//             j0 = shell_to_function[j_shell] - shell_to_function[j_shell_begin];
//             if (dimension != 0)
//             {
//                 xyz_offset_for_i_shell = atm(PTR_COORD, bas(ATOM_OF, i_shell));
//                 shift_bas(env_loc, env, vectors_to_nearest_images, xyz_offset_for_i_shell, m_image);
//             }
//             _apply_rho(density_kernel, rho_priv, density_matrices + m_image * n_ao_pairs + j0 * n_ao_i + i0,
//                        dims, n_components, precision_in_log, dimension, lattice_vector, reciprocal_lattice_vector,
//                        offset, lattice_sum_mesh, global_mesh, shell_indices,
//                        atm, n_atoms, bas, n_basis, env_loc, cache);
//         }
//         NPomp_dsum_reduce_inplace(density_buffer, n_components * n_grid_points_for_summation);
//         free(cache);
//         free(env_loc);
//         if (thread_id != 0)
//         {
//             free(rho_priv);
//         }
//     }
// }
