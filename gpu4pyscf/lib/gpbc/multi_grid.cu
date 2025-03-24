
#include <cub/cub.cuh>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>

#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]

#define EIJ_CUTOFF 60

__constant__ double lattice_vectors[9];
__constant__ double reciprocal_lattice_vectors[9];
__constant__ double dxyz_dabc[9];
__constant__ double reciprocal_norm[3];

__host__ __device__ double distance_squared(const double x, const double y,
                                            const double z) {
  return x * x + y * y + z * z;
}

template <int ANG> __inline__ __device__ constexpr double common_fac_sp() {
  if constexpr (ANG == 0) {
    return 0.282094791773878143;
  } else if constexpr (ANG == 1) {
    return 0.488602511902919921;
  } else {
    return 1.0;
  }
}

inline __device__ void gto_cartesian_s(double values[], double fx, double fy,
                                       double fz) {
  values[0] = 1;
}

inline __device__ void gto_cartesian_p(double values[], double fx, double fy,
                                       double fz) {
  values[0] = fx;
  values[1] = fy;
  values[2] = fz;
}

inline __device__ void gto_cartesian_d(double values[], double fx, double fy,
                                       double fz) {
  values[0] = fx * fx;
  values[1] = fx * fy;
  values[2] = fx * fz;
  values[3] = fy * fy;
  values[4] = fy * fz;
  values[5] = fz * fz;
}

inline __device__ void gto_cartesian_f(double values[], double fx, double fy,
                                       double fz) {
  values[0] = fx * fx * fx;
  values[1] = fx * fx * fy;
  values[2] = fx * fx * fz;
  values[3] = fx * fy * fy;
  values[4] = fx * fy * fz;
  values[5] = fx * fz * fz;
  values[6] = fy * fy * fy;
  values[7] = fy * fy * fz;
  values[8] = fy * fz * fz;
  values[9] = fz * fz * fz;
}

inline __device__ void gto_cartesian_g(double values[], double fx, double fy,
                                       double fz) {
  values[0] = fx * fx * fx * fx;
  values[1] = fx * fx * fx * fy;
  values[2] = fx * fx * fx * fz;
  values[3] = fx * fx * fy * fy;
  values[4] = fx * fx * fy * fz;
  values[5] = fx * fx * fz * fz;
  values[6] = fx * fy * fy * fy;
  values[7] = fx * fy * fy * fz;
  values[8] = fx * fy * fz * fz;
  values[9] = fx * fz * fz * fz;
  values[10] = fy * fy * fy * fy;
  values[11] = fy * fy * fy * fz;
  values[12] = fy * fy * fz * fz;
  values[13] = fy * fz * fz * fz;
  values[14] = fz * fz * fz * fz;
}

template <int ANG>
inline __device__ void gto_cartesian(double values[], double fx, double fy,
                                     double fz) {
  if constexpr (ANG == 0) {
    gto_cartesian_s(values, fx, fy, fz);
  } else if constexpr (ANG == 1) {
    gto_cartesian_p(values, fx, fy, fz);
  } else if constexpr (ANG == 2) {
    gto_cartesian_d(values, fx, fy, fz);
  } else if constexpr (ANG == 3) {
    gto_cartesian_f(values, fx, fy, fz);
  } else if constexpr (ANG == 4) {
    gto_cartesian_g(values, fx, fy, fz);
  }
}

template <int n_channels, int i_angular, int j_angular, int block_dim_a,
          int block_dim_b, int block_dim_c>
__global__ void evaluate_density_kernel(
    double *density, const double *density_matrices,
    const int *non_trivial_pairs, const double *cutoffs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = block_dim_a * block_dim_b * block_dim_c;

  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride = density_matrix_stride * n_images;

  const int block_index = sorted_block_index[blockIdx.x];
  const int n_blocks_b = (mesh_b + block_dim_b - 1) / block_dim_b;
  const int n_blocks_c = (mesh_c + block_dim_c - 1) / block_dim_c;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * block_dim_a;
  const int b_start = block_b_index * block_dim_b;
  const int c_start = block_c_index * block_dim_c;

  const double start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const double start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const double start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + block_dim_a, mesh_a) - a_start;
  const int b_upper = min(b_start + block_dim_b, mesh_b) - b_start;
  const int c_upper = min(c_start + block_dim_c, mesh_c) - c_start;

  const int thread_id = threadIdx.x + threadIdx.y * block_dim_c +
                        threadIdx.z * block_dim_c * block_dim_b;

  double i_cartesian[n_i_cartesian_functions];
  double j_cartesian[n_j_cartesian_functions];

  double
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

  __shared__ double reduced_density_values[n_channels * n_threads];

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    reduced_density_values[i_channel * n_threads + thread_id] = 0;
  }

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  int a_index, b_index, c_index;
  double x, y, z;

  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const double i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const double i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const double i_y = env[i_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_i * 3 + 1];
    const double i_z = env[i_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_i * 3 + 2];
    const double i_coeff = env[bas(PTR_COEFF, i_shell)];

    const double j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const double j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const double j_y = env[j_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_j * 3 + 1];
    const double j_z = env[j_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_j * 3 + 2];
    const double j_coeff = env[bas(PTR_COEFF, j_shell)];

    const double ij_exponent = i_exponent + j_exponent;
    const double ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const double gaussian_exponent_at_reference =
        ij_exponent * distance_squared(start_position_x - pair_x,
                                       start_position_y - pair_y,
                                       start_position_z - pair_z);

    const double pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<i_angular>() *
                  common_fac_sp<j_angular>()
            : 0;
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          const double density_matrix_value =
              density_matrices[density_matrix_channel_stride * i_channel +
                               (i_function + i_function_index) * n_j_functions +
                               j_function + j_function_index];

          prefactor[i_channel * n_i_cartesian_functions *
                        n_j_cartesian_functions +
                    i_function_index * n_j_cartesian_functions +
                    j_function_index] = pair_prefactor * density_matrix_value;
        }
      }
    }

    // From now on we assume that the lattice is orthogonal.
    // Shouldn't be too hard to extend to non-orthogonal lattices.

    const double exp_cross_term_a =
        exp(-2 * ij_exponent * dxyz_dabc[0] * (start_position_x - pair_x));
    const double exp_cross_term_b =
        exp(-2 * ij_exponent * dxyz_dabc[4] * (start_position_y - pair_y));
    const double exp_cross_term_c =
        exp(-2 * ij_exponent * dxyz_dabc[8] * (start_position_z - pair_z));
    const double exp_da_squared =
        exp(-ij_exponent * dxyz_dabc[0] * dxyz_dabc[0]);
    const double exp_db_squared =
        exp(-ij_exponent * dxyz_dabc[4] * dxyz_dabc[4]);
    const double exp_dc_squared =
        exp(-ij_exponent * dxyz_dabc[8] * dxyz_dabc[8]);

    double gaussian_x, gaussian_y, gaussian_z, exp_n_dx_squared,
        exp_n_dy_squared, exp_n_dz_squared;
    for (a_index = 0, gaussian_x = 1, exp_n_dx_squared = exp_da_squared,
        x = start_position_x;
         a_index < a_upper;
         a_index++, gaussian_x *= exp_n_dx_squared * exp_cross_term_a,
        exp_n_dx_squared *= exp_da_squared * exp_da_squared,
        x += dxyz_dabc[0]) {
      for (b_index = 0, gaussian_y = 1, exp_n_dy_squared = exp_db_squared,
          y = start_position_y;
           b_index < b_upper;
           b_index++, gaussian_y *= exp_n_dy_squared * exp_cross_term_b,
          exp_n_dy_squared *= exp_db_squared * exp_db_squared,
          y += dxyz_dabc[4]) {
        for (c_index = 0, gaussian_z = 1, exp_n_dz_squared = exp_dc_squared,
            z = start_position_z;
             c_index < c_upper;
             c_index++, gaussian_z *= exp_n_dz_squared * exp_cross_term_c,
            exp_n_dz_squared *= exp_dc_squared * exp_dc_squared,
            z += dxyz_dabc[8]) {
          gto_cartesian<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
          gto_cartesian<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);

          const double gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            double density_value_to_be_shared = 0;
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                density_value_to_be_shared +=
                    prefactor[i_channel * n_i_cartesian_functions *
                                  n_j_cartesian_functions +
                              i_function_index * n_j_cartesian_functions +
                              j_function_index] *
                    i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }

            density_value_to_be_shared *= gaussian;

            double reduced =
                cub::BlockReduce<double, block_dim_c,
                                 cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                                 block_dim_b, block_dim_a>()
                    .Sum(density_value_to_be_shared);
            if (thread_id == 0) {
              reduced_density_values[i_channel * n_threads +
                                     a_index * block_dim_b * block_dim_c +
                                     b_index * block_dim_c + c_index] +=
                  reduced;
            }
          }
        }
      }
    }
  }
  a_index = a_start + threadIdx.z;
  b_index = b_start + threadIdx.y;
  c_index = c_start + threadIdx.x;

  __syncthreads();

  if (a_index < mesh_a && b_index < mesh_b && c_index < mesh_c) {
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      atomicAdd(density + i_channel * mesh_a * mesh_b * mesh_c +
                    a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
                reduced_density_values[i_channel * n_threads + thread_id]);
    }
  }
}

#define density_kernel_macro(li, lj)                                           \
  evaluate_density_kernel<n_channels, li, lj, block_dim_a, block_dim_b,        \
                          block_dim_c><<<block_grid, block_size>>>(            \
      density, density_matrices, non_trivial_pairs, cutoffs, i_shells,         \
      j_shells, n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions, \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      sorted_block_index, image_indices, vectors_to_neighboring_images,        \
      n_images, mesh_a, mesh_b, mesh_c, atm, bas, env)

#define density_kernel_case_macro(li, lj)                                      \
  case (li * 10 + lj):                                                         \
    density_kernel_macro(li, lj);                                              \
    break

template <int n_channels, int block_dim_a, int block_dim_b, int block_dim_c>
void evaluate_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *mesh, const int *atm, const int *bas,
    const double *env) {
  dim3 block_size(block_dim_c, block_dim_b, block_dim_a);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);
  switch (i_angular * 10 + j_angular) {
    density_kernel_case_macro(0, 0);
    density_kernel_case_macro(0, 1);
    density_kernel_case_macro(0, 2);
    density_kernel_case_macro(0, 3);
    density_kernel_case_macro(0, 4);
    density_kernel_case_macro(1, 0);
    density_kernel_case_macro(1, 1);
    density_kernel_case_macro(1, 2);
    density_kernel_case_macro(1, 3);
    density_kernel_case_macro(1, 4);
    density_kernel_case_macro(2, 0);
    density_kernel_case_macro(2, 1);
    density_kernel_case_macro(2, 2);
    density_kernel_case_macro(2, 3);
    density_kernel_case_macro(2, 4);
    density_kernel_case_macro(3, 0);
    density_kernel_case_macro(3, 1);
    density_kernel_case_macro(3, 2);
    density_kernel_case_macro(3, 3);
    density_kernel_case_macro(3, 4);
    density_kernel_case_macro(4, 0);
    density_kernel_case_macro(4, 1);
    density_kernel_case_macro(4, 2);
    density_kernel_case_macro(4, 3);
    density_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_density_driver\n",
            i_angular, j_angular);
  }

  checkCudaErrors(cudaPeekAtLastError());
}

template <int n_channels, int i_angular, int j_angular, int block_dim_a,
          int block_dim_b, int block_dim_c>
__global__ void evaluate_xc_kernel(
    double *fock, const double *xc_weights, const int *non_trivial_pairs,
    const double *cutoffs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = block_dim_a * block_dim_b * block_dim_c;

  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int fock_stride = n_i_functions * n_j_functions;

  const int block_index = sorted_block_index[blockIdx.x];

  const int n_blocks_b = (mesh_b + block_dim_b - 1) / block_dim_b;
  const int n_blocks_c = (mesh_c + block_dim_c - 1) / block_dim_c;

  const int block_a_stride = n_blocks_b * n_blocks_c;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_blocks_c;
  const int block_c_index = block_ab_index % n_blocks_c;

  const int a_start = block_a_index * block_dim_a;
  const int b_start = block_b_index * block_dim_b;
  const int c_start = block_c_index * block_dim_c;

  const double start_position_x =
      dxyz_dabc[0] * a_start + dxyz_dabc[3] * b_start + dxyz_dabc[6] * c_start;
  const double start_position_y =
      dxyz_dabc[1] * a_start + dxyz_dabc[4] * b_start + dxyz_dabc[7] * c_start;
  const double start_position_z =
      dxyz_dabc[2] * a_start + dxyz_dabc[5] * b_start + dxyz_dabc[8] * c_start;

  const int a_upper = min(a_start + block_dim_a, mesh_a) - a_start;
  const int b_upper = min(b_start + block_dim_b, mesh_b) - b_start;
  const int c_upper = min(c_start + block_dim_c, mesh_c) - c_start;

  double neighboring_gaussian_sum[n_channels * n_i_cartesian_functions *
                                  n_j_cartesian_functions];

  double i_cartesian[n_i_cartesian_functions];
  double j_cartesian[n_j_cartesian_functions];

  const int start_pair_index = accumulated_n_pairs_per_local_grid[block_index];
  const int end_pair_index =
      accumulated_n_pairs_per_local_grid[block_index + 1];
  const int n_pairs = end_pair_index - start_pair_index;
  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  __shared__ double
      xc_values[n_channels * block_dim_a * block_dim_b * block_dim_c];

  int a_index = a_start + threadIdx.z;
  int b_index = b_start + threadIdx.y;
  int c_index = c_start + threadIdx.x;

  const bool out_of_boundary =
      a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c;
  double xc_value = 0;

  const int thread_id = threadIdx.x + threadIdx.y * block_dim_c +
                        threadIdx.z * block_dim_c * block_dim_b;

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    if (!out_of_boundary) {
      xc_value =
          xc_weights[i_channel * xc_weights_stride + a_index * mesh_b * mesh_c +
                     b_index * mesh_c + c_index];
    }

    xc_values[i_channel * block_dim_a * block_dim_b * block_dim_c + thread_id] =
        xc_value;
  }
  __syncthreads();
  double x, y, z;
  for (int i_batch = 0, i_pair_index = start_pair_index + thread_id;
       i_batch < n_batches; i_batch++, i_pair_index += n_threads) {
    const bool is_valid_pair = i_pair_index < end_pair_index;
    const int i_pair =
        is_valid_pair ? sorted_pairs_per_local_grid[i_pair_index] : 0;

    const int image_index = image_indices[i_pair];
    const int image_index_i = image_index / n_images;
    const int image_index_j = image_index % n_images;

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
    const int i_shell = i_shells[i_shell_index];
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_shell = j_shells[j_shell_index];
    const int j_function = shell_to_ao_indices[j_shell];

    const double i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const double i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index_i * 3];
    const double i_y = env[i_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_i * 3 + 1];
    const double i_z = env[i_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_i * 3 + 2];
    const double i_coeff = env[bas(PTR_COEFF, i_shell)];

    const double j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const double j_x =
        env[j_coord_offset] + vectors_to_neighboring_images[image_index_j * 3];
    const double j_y = env[j_coord_offset + 1] +
                       vectors_to_neighboring_images[image_index_j * 3 + 1];
    const double j_z = env[j_coord_offset + 2] +
                       vectors_to_neighboring_images[image_index_j * 3 + 2];
    const double j_coeff = env[bas(PTR_COEFF, j_shell)];

    const double ij_exponent = i_exponent + j_exponent;
    const double ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;
    const double gaussian_exponent_at_reference =
        ij_exponent * distance_squared(start_position_x - pair_x,
                                       start_position_y - pair_y,
                                       start_position_z - pair_z);

    const double pair_prefactor =
        is_valid_pair
            ? exp(-ij_exponent_in_prefactor - gaussian_exponent_at_reference) *
                  i_coeff * j_coeff * common_fac_sp<i_angular>() *
                  common_fac_sp<j_angular>()
            : 0;

    const double exp_cross_term_a =
        exp(-2 * ij_exponent * dxyz_dabc[0] * (start_position_x - pair_x));
    const double exp_cross_term_b =
        exp(-2 * ij_exponent * dxyz_dabc[4] * (start_position_y - pair_y));
    const double exp_cross_term_c =
        exp(-2 * ij_exponent * dxyz_dabc[8] * (start_position_z - pair_z));

    const double exp_da_squared =
        exp(-ij_exponent * dxyz_dabc[0] * dxyz_dabc[0]);
    const double exp_db_squared =
        exp(-ij_exponent * dxyz_dabc[4] * dxyz_dabc[4]);
    const double exp_dc_squared =
        exp(-ij_exponent * dxyz_dabc[8] * dxyz_dabc[8]);

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
#pragma unroll
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                       n_j_cartesian_functions +
                                   i_function_index * n_j_cartesian_functions +
                                   j_function_index] = 0;
        }
      }
    }
    double gaussian_x, gaussian_y, gaussian_z, exp_n_dx_squared,
        exp_n_dy_squared, exp_n_dz_squared;

    for (a_index = 0, gaussian_x = 1, exp_n_dx_squared = exp_da_squared,
        x = start_position_x;
         a_index < a_upper;
         a_index++, gaussian_x *= exp_n_dx_squared * exp_cross_term_a,
        exp_n_dx_squared *= exp_da_squared * exp_da_squared,
        x += dxyz_dabc[0]) {
      for (b_index = 0, gaussian_y = 1, exp_n_dy_squared = exp_db_squared,
          y = start_position_y;
           b_index < b_upper;
           b_index++, gaussian_y *= exp_n_dy_squared * exp_cross_term_b,
          exp_n_dy_squared *= exp_db_squared * exp_db_squared,
          y += dxyz_dabc[4]) {
        for (c_index = 0, gaussian_z = 1, exp_n_dz_squared = exp_dc_squared,
            z = start_position_z;
             c_index < c_upper;
             c_index++, gaussian_z *= exp_n_dz_squared * exp_cross_term_c,
            exp_n_dz_squared *= exp_dc_squared * exp_dc_squared,
            z += dxyz_dabc[8]) {
          gto_cartesian<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
          gto_cartesian<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);

          const double gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            xc_value =
                gaussian *
                xc_values[i_channel * block_dim_a * block_dim_b * block_dim_c +
                          a_index * block_dim_b * block_dim_c +
                          b_index * block_dim_c + c_index];
#pragma unroll
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
#pragma unroll
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                             n_j_cartesian_functions +
                                         i_function_index *
                                             n_j_cartesian_functions +
                                         j_function_index] +=
                    xc_value * i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }
          }
        }
      }
    }

    if (is_valid_pair) {
      double *fock_pointer = fock + i_function * n_j_functions + j_function;

#pragma unroll
      for (int i_channel = 0; i_channel < n_channels; i_channel++) {
#pragma unroll
        for (int i_function_index = 0;
             i_function_index < n_i_cartesian_functions; i_function_index++) {
#pragma unroll
          for (int j_function_index = 0;
               j_function_index < n_j_cartesian_functions; j_function_index++) {
            atomicAdd(
                fock_pointer,
                neighboring_gaussian_sum[i_channel * n_i_cartesian_functions *
                                             n_j_cartesian_functions +
                                         i_function_index *
                                             n_j_cartesian_functions +
                                         j_function_index] *
                    pair_prefactor);
            fock_pointer++;
          }
          fock_pointer += n_j_functions - n_j_cartesian_functions;
        }
        fock_pointer +=
            n_images * fock_stride - n_i_cartesian_functions * n_j_functions;
      }
    }
  }
}

#define xc_kernel_macro(li, lj)                                                \
  evaluate_xc_kernel<n_channels, li, lj, block_dim_a, block_dim_b,             \
                     block_dim_c><<<block_grid, block_size>>>(                 \
      fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, j_shells,        \
      n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,           \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      sorted_block_index, image_indices, vectors_to_neighboring_images,        \
      n_images, mesh_a, mesh_b, mesh_c, atm, bas, env)

#define xc_kernel_case_macro(li, lj)                                           \
  case (li * 10 + lj):                                                         \
    xc_kernel_macro(li, lj);                                                   \
    break

template <int n_channels, int block_dim_a, int block_dim_b, int block_dim_c>
void evaluate_xc_driver(
    double *fock, const double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *mesh, const int *atm, const int *bas,
    const double *env) {
  dim3 block_size(block_dim_c, block_dim_b, block_dim_a);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid(n_contributing_blocks, 1, 1);

  switch (i_angular * 10 + j_angular) {
    xc_kernel_case_macro(0, 0);
    xc_kernel_case_macro(0, 1);
    xc_kernel_case_macro(0, 2);
    xc_kernel_case_macro(0, 3);
    xc_kernel_case_macro(0, 4);
    xc_kernel_case_macro(1, 0);
    xc_kernel_case_macro(1, 1);
    xc_kernel_case_macro(1, 2);
    xc_kernel_case_macro(1, 3);
    xc_kernel_case_macro(1, 4);
    xc_kernel_case_macro(2, 0);
    xc_kernel_case_macro(2, 1);
    xc_kernel_case_macro(2, 2);
    xc_kernel_case_macro(2, 3);
    xc_kernel_case_macro(2, 4);
    xc_kernel_case_macro(3, 0);
    xc_kernel_case_macro(3, 1);
    xc_kernel_case_macro(3, 2);
    xc_kernel_case_macro(3, 3);
    xc_kernel_case_macro(3, 4);
    xc_kernel_case_macro(4, 0);
    xc_kernel_case_macro(4, 1);
    xc_kernel_case_macro(4, 2);
    xc_kernel_case_macro(4, 3);
    xc_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_xc_driver\n",
            i_angular, j_angular);
  }

  checkCudaErrors(cudaPeekAtLastError());
}

extern "C" {
void update_lattice_vectors(const double *lattice_vectors_on_device,
                            const double *reciprocal_lattice_vectors_on_device,
                            const double *reciprocal_norm_on_device) {
  cudaMemcpyToSymbol(lattice_vectors, lattice_vectors_on_device,
                     9 * sizeof(double));
  cudaMemcpyToSymbol(reciprocal_lattice_vectors,
                     reciprocal_lattice_vectors_on_device, 9 * sizeof(double));
  cudaMemcpyToSymbol(reciprocal_norm, reciprocal_norm_on_device,
                     3 * sizeof(double));
}

void update_dxyz_dabc(const double *dxyz_dabc_on_device) {
  cudaMemcpyToSymbol(dxyz_dabc, dxyz_dabc_on_device, 9 * sizeof(double));
}

void evaluate_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_blocks, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int *mesh, const int *atm, const int *bas, const double *env,
    const int blocking_sizes[3], const int n_channels) {
  if (blocking_sizes[0] == 4 && blocking_sizes[1] == 4 &&
      blocking_sizes[2] == 4) {
    if (n_channels == 1) {
      evaluate_density_driver<1, 4, 4, 4>(
          density, density_matrices, i_angular, j_angular, non_trivial_pairs,
          cutoffs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
          n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
          accumulated_n_pairs_per_local_grid, sorted_block_index, n_blocks,
          image_indices, vectors_to_neighboring_images, n_images, mesh, atm,
          bas, env);
    } else if (n_channels == 2) {
      evaluate_density_driver<2, 4, 4, 4>(
          density, density_matrices, i_angular, j_angular, non_trivial_pairs,
          cutoffs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
          n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
          accumulated_n_pairs_per_local_grid, sorted_block_index, n_blocks,
          image_indices, vectors_to_neighboring_images, n_images, mesh, atm,
          bas, env);
    } else {
      fprintf(stderr, "n_channels more than 2 is not supported.\n");
    }
  } else {
    fprintf(stderr, "blocking_sizes other than 4 is not supported.\n");
  }
}

void evaluate_xc_driver(
    double *fock, const double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid,
    const int *sorted_block_index, const int n_contributing_blocks,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const int *mesh, const int *atm, const int *bas,
    const double *env, const int blocking_sizes[3], const int n_channels) {
  if (blocking_sizes[0] == 4 && blocking_sizes[1] == 4 &&
      blocking_sizes[2] == 4) {
    if (n_channels == 1) {
      evaluate_xc_driver<1, 4, 4, 4>(
          fock, xc_weights, i_angular, j_angular, non_trivial_pairs, cutoffs,
          i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
          n_j_functions, sorted_pairs_per_local_grid,
          accumulated_n_pairs_per_local_grid, sorted_block_index,
          n_contributing_blocks, image_indices, vectors_to_neighboring_images,
          n_images, mesh, atm, bas, env);
    } else if (n_channels == 2) {
      evaluate_xc_driver<2, 4, 4, 4>(
          fock, xc_weights, i_angular, j_angular, non_trivial_pairs, cutoffs,
          i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
          n_j_functions, sorted_pairs_per_local_grid,
          accumulated_n_pairs_per_local_grid, sorted_block_index,
          n_contributing_blocks, image_indices, vectors_to_neighboring_images,
          n_images, mesh, atm, bas, env);
    } else {
      fprintf(stderr, "n_channels more than 2 is not supported.\n");
    }
  } else {
    fprintf(stderr, "blocking_sizes other than 4 is not supported.\n");
  }
}
}