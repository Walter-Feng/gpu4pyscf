#include <cstdio>
#include <cuda_runtime.h>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>

#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]

#define EIJ_CUTOFF 60

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

template <typename T, unsigned int blockSize>
__device__ void warpReduce(volatile T *sdata, unsigned int tid) {
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <typename T, unsigned int blockSize>
__device__ void reduce_and_add(volatile T *shared_memory, T *output,
                               unsigned int thread_id, T number_to_be_added) {
  __syncthreads();
  shared_memory[thread_id] = number_to_be_added;
  __syncthreads();
  if constexpr (blockSize >= 512) {
    if (thread_id < 256)
      shared_memory[thread_id] += shared_memory[thread_id + 256];
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (thread_id < 128)
      shared_memory[thread_id] += shared_memory[thread_id + 128];
    __syncthreads();
  }
  if constexpr (blockSize >= 128) {
    if (thread_id < 64)
      shared_memory[thread_id] += shared_memory[thread_id + 64];
    __syncthreads();
  }
  if (thread_id < 32)
    warpReduce<T, blockSize>(shared_memory, thread_id);
  if (thread_id == 0)
    atomicAdd(output, shared_memory[0]);
}

template <int ANG>
__device__ static void gto_cartesian(double *g, double fx, double fy,
                                     double fz) {
  for (int lx = ANG, i = 0; lx >= 0; lx--) {
    for (int ly = ANG - lx; ly >= 0; ly--, i++) {
      int lz = ANG - lx - ly;
      g[i] = pow(fx, lx) * pow(fy, ly) * pow(fz, lz);
    }
  }
}

template <int n_channels, int i_angular, int j_angular>
__global__ void evaluate_density_kernel(
    double *density, const double *density_matrices,
    const int *non_trivial_pairs, const double *cutoffs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {
  const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
  const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
  const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride = density_matrix_stride * n_images;

  if (a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c) {
    return;
  }

  const double position_x = lattice_vectors[0] * a_index / mesh_a +
                            lattice_vectors[3] * b_index / mesh_b +
                            lattice_vectors[6] * c_index / mesh_c;
  const double position_y = lattice_vectors[1] * a_index / mesh_a +
                            lattice_vectors[4] * b_index / mesh_b +
                            lattice_vectors[7] * c_index / mesh_c;
  const double position_z = lattice_vectors[2] * a_index / mesh_a +
                            lattice_vectors[5] * b_index / mesh_b +
                            lattice_vectors[8] * c_index / mesh_c;

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;

  double i_cartesian[n_i_cartesian_functions];
  double j_cartesian[n_j_cartesian_functions];

  double density_value[n_channels];
  double
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    density_value[i_channel] = 0;
  }

  const uint local_grid_index =
      blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

  for (int i_pair_index = accumulated_n_pairs_per_local_grid[local_grid_index];
       i_pair_index < accumulated_n_pairs_per_local_grid[local_grid_index + 1];
       i_pair_index++) {

    const int i_pair = sorted_pairs_per_local_grid[i_pair_index];
    const int image_index = image_indices[i_pair];

    const double cutoff = cutoffs[i_pair];
    const double image_x = vectors_to_neighboring_images[image_index * 3];
    const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
    const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
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
    const double ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff *
                                  j_coeff * common_fac_sp<i_angular>() *
                                  common_fac_sp<j_angular>();
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          const double density_matrix_value =
              density_matrices[density_matrix_channel_stride * i_channel +
                               image_index * density_matrix_stride +
                               (i_function + i_function_index) * n_j_functions +
                               j_function + j_function_index];

          prefactor[i_channel * n_i_cartesian_functions *
                        n_j_cartesian_functions +
                    i_function_index * n_j_cartesian_functions +
                    j_function_index] = pair_prefactor * density_matrix_value;
        }
      }
    }

    const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const int lower_a_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[0] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[1] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[2]);
    const int upper_a_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[0] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[1] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[2]);

    const int lower_b_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[3] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[4] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[5]);
    const int upper_b_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[3] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[4] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[5]);

    const int lower_c_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[6] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[7] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[8]);
    const int upper_c_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[6] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[7] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[8]);

    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
      for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
        for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {

          const double x = position_x - a_cell * lattice_vectors[0] -
                           b_cell * lattice_vectors[3] -
                           c_cell * lattice_vectors[6];
          const double y = position_y - a_cell * lattice_vectors[1] -
                           b_cell * lattice_vectors[4] -
                           c_cell * lattice_vectors[7];
          const double z = position_z - a_cell * lattice_vectors[2] -
                           b_cell * lattice_vectors[5] -
                           c_cell * lattice_vectors[8];
          gto_cartesian<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
          gto_cartesian<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);

          const double r_squared =
              distance_squared(x - pair_x, y - pair_y, z - pair_z);
          const double gaussian = exp(-ij_exponent * r_squared);

#pragma unroll
          for (int i_channel = 0; i_channel < n_channels; i_channel++) {
            for (int i_function_index = 0;
                 i_function_index < n_i_cartesian_functions;
                 i_function_index++) {
              for (int j_function_index = 0;
                   j_function_index < n_j_cartesian_functions;
                   j_function_index++) {
                density_value[i_channel] +=
                    prefactor[i_channel * n_i_cartesian_functions *
                                  n_j_cartesian_functions +
                              i_function_index * n_j_cartesian_functions +
                              j_function_index] *
                    gaussian * i_cartesian[i_function_index] *
                    j_cartesian[j_function_index];
              }
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    atomicAdd(density + i_channel * mesh_a * mesh_b * mesh_c +
                  a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
              density_value[i_channel]);
  }
}

#define density_kernel_macro(li, lj)                                           \
  evaluate_density_kernel<n_channels, li, lj><<<block_grid, block_size>>>(     \
      density, density_matrices, non_trivial_pairs, cutoffs, i_shells,         \
      j_shells, n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions, \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      image_indices, vectors_to_neighboring_images, n_images, lattice_vectors, \
      reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env)

#define density_kernel_case_macro(li, lj)                                      \
  case (li * 10 + lj):                                                         \
    density_kernel_macro(li, lj);                                              \
    break

template <int n_channels>
void evaluate_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int *mesh, const int *atm, const int *bas, const double *env,
    const int blocking_sizes[3]) {
  dim3 block_size(blocking_sizes[2], blocking_sizes[1], blocking_sizes[0]);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid((mesh_c + blocking_sizes[2] - 1) / blocking_sizes[2],
                  (mesh_b + blocking_sizes[1] - 1) / blocking_sizes[1],
                  (mesh_a + blocking_sizes[0] - 1) / blocking_sizes[0]);
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

template <int n_channels, int i_angular, int j_angular>
__global__ void evaluate_diffused_density_kernel(
    double *density, const double *density_matrices,
    const int *non_trivial_pairs, const int n_pairs, const int *i_shells,
    const int *j_shells, const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *contributing_area_begin, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int mesh_a, const int mesh_b, const int mesh_c, const int a_range,
    const int b_range, const int c_range, const int *atm, const int *bas,
    const double *env) {

  const uint i_pair = threadIdx.x + blockDim.x * blockIdx.x;
  if (i_pair >= n_pairs) {
    return;
  }

  const int a_stride = mesh_b * mesh_c;
  const int b_stride = mesh_c;
  const int channel_stride = mesh_a * a_stride;
  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  const int density_matrix_stride = n_i_functions * n_j_functions;
  const int density_matrix_channel_stride = density_matrix_stride * n_images;
  double i_cartesian[n_i_cartesian_functions];
  double j_cartesian[n_j_cartesian_functions];
  const int image_index = image_indices[i_pair];

  const double image_x = vectors_to_neighboring_images[image_index * 3];
  const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
  const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

  const int shell_pair_index = non_trivial_pairs[i_pair];
  const int i_shell_index = shell_pair_index / n_j_shells;
  const int j_shell_index = shell_pair_index % n_j_shells;
  const int i_shell = i_shells[i_shell_index];
  const int i_function = shell_to_ao_indices[i_shell];
  const int j_shell = j_shells[j_shell_index];
  const int j_function = shell_to_ao_indices[j_shell];

  double
      prefactor[n_channels * n_i_cartesian_functions * n_j_cartesian_functions];

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
  const double ij_exponent_in_prefactor =
      i_exponent * j_exponent / ij_exponent *
      distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

  const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff *
                                j_coeff * common_fac_sp<i_angular>() *
                                common_fac_sp<j_angular>();

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
         i_function_index++) {
      for (int j_function_index = 0; j_function_index < n_j_cartesian_functions;
           j_function_index++) {
        const double density_matrix_value =
            density_matrices[density_matrix_channel_stride * i_channel +
                             image_index * density_matrix_stride +
                             (i_function + i_function_index) * n_j_functions +
                             j_function + j_function_index];

        prefactor[i_channel * n_i_cartesian_functions *
                      n_j_cartesian_functions +
                  i_function_index * n_j_cartesian_functions +
                  j_function_index] = pair_prefactor * density_matrix_value;
      }
    }
  }

  const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
  const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
  const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

  const int a0 = contributing_area_begin[i_pair];
  const int b0 = contributing_area_begin[n_pairs + i_pair];
  const int c0 = contributing_area_begin[2 * n_pairs + i_pair];

  const double x0 = lattice_vectors[0] * a0 / mesh_a +
                    lattice_vectors[3] * b0 / mesh_b +
                    lattice_vectors[6] * c0 / mesh_c - pair_x;
  const double y0 = lattice_vectors[1] * a0 / mesh_a +
                    lattice_vectors[4] * b0 / mesh_b +
                    lattice_vectors[7] * c0 / mesh_c - pair_y;
  const double z0 = lattice_vectors[2] * a0 / mesh_a +
                    lattice_vectors[5] * b0 / mesh_b +
                    lattice_vectors[8] * c0 / mesh_c - pair_z;

  const int a_start = mesh_a + a0 % mesh_a;
  const int b_start = mesh_b + b0 % mesh_b;
  const int c_start = mesh_c + c0 % mesh_c;

  // asserting orthogonal lattice vectors

  const double gaussian_x0 = exp(-ij_exponent * x0 * x0);
  const double gaussian_y0 = exp(-ij_exponent * y0 * y0);
  const double gaussian_z0 = exp(-ij_exponent * z0 * z0);

  const double dx = lattice_vectors[0] / mesh_a;
  const double dy = lattice_vectors[4] / mesh_b;
  const double dz = lattice_vectors[8] / mesh_c;

  const double exp_cross_term_x = exp(-2 * ij_exponent * dx * x0);
  const double exp_cross_term_y = exp(-2 * ij_exponent * dy * y0);
  const double exp_cross_term_z = exp(-2 * ij_exponent * dz * z0);
  const double exp_dx_squared = exp(-ij_exponent * dx * dx);
  const double exp_dy_squared = exp(-ij_exponent * dy * dy);
  const double exp_dz_squared = exp(-ij_exponent * dz * dz);
  const double exp_2dx_squared = exp_dx_squared * exp_dx_squared;
  const double exp_2dy_squared = exp_dy_squared * exp_dy_squared;
  const double exp_2dz_squared = exp_dz_squared * exp_dz_squared;

  double exp_n_dx_squared = exp_dx_squared;
  double gaussian_x = gaussian_x0;
  double x = x0;
  int a_index = a_start;
  for (int a = 0; a <= a_range; a++) {
    double exp_n_dy_squared = exp_dy_squared;
    double gaussian_y = gaussian_y0;
    double y = y0;
    int b_index = b_start;
    for (int b = 0; b <= b_range; b++) {
      double exp_n_dz_squared = exp_dz_squared;
      double gaussian_z = gaussian_z0;
      double z = z0;
      int c_index = c_start;
      for (int c = 0; c <= c_range; c++) {
        gto_cartesian<i_angular>(i_cartesian, x, y, z);
        gto_cartesian<j_angular>(j_cartesian, x, y, z);
        const double gaussian = gaussian_x * gaussian_y * gaussian_z;
#pragma unroll
        for (int i_channel = 0; i_channel < n_channels; i_channel++) {
          double density_value = 0;
          for (int i_function_index = 0;
               i_function_index < n_i_cartesian_functions; i_function_index++) {
            for (int j_function_index = 0;
                 j_function_index < n_j_cartesian_functions;
                 j_function_index++) {
              density_value +=
                  prefactor[i_channel * n_i_cartesian_functions *
                                n_j_cartesian_functions +
                            i_function_index * n_j_cartesian_functions +
                            j_function_index] *
                  gaussian * i_cartesian[i_function_index] *
                  j_cartesian[j_function_index];
            }
          }
          atomicAdd(density + i_channel * channel_stride + a_index * a_stride +
                        b_index * b_stride + c_index,
                    density_value);
        }
        gaussian_z *= exp_n_dz_squared * exp_cross_term_z;
        exp_n_dz_squared *= exp_2dz_squared;
        z += dz;
        c_index++;
        if (c_index == mesh_c) {
          c_index = 0;
        }
      }
      gaussian_y *= exp_n_dy_squared * exp_cross_term_y;
      exp_n_dy_squared *= exp_2dy_squared;
      y += dy;
      b_index++;
      if (b_index == mesh_b) {
        b_index = 0;
      }
    }
    gaussian_x *= exp_n_dx_squared * exp_cross_term_x;
    exp_n_dx_squared *= exp_2dx_squared;
    x += dx;
    a_index++;
    if (a_index == mesh_a) {
      a_index = 0;
    }
  }
}

#define diffused_density_kernel_macro(li, lj)                                  \
  evaluate_diffused_density_kernel<n_channels, li, lj>                         \
      <<<block_grid, block_size>>>(                                            \
          density, density_matrices, non_trivial_pairs, n_pairs, i_shells,     \
          j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,            \
          n_j_functions, contributing_area_begin, image_indices,               \
          vectors_to_neighboring_images, n_images, lattice_vectors,            \
          reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, a_range,         \
          b_range, c_range, atm, bas, env)

#define diffused_density_kernel_case_macro(li, lj)                             \
  case (li * 10 + lj):                                                         \
    diffused_density_kernel_macro(li, lj);                                     \
    break

template <int n_channels>
void evaluate_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int n_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *contributing_area_begin,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const double *lattice_vectors,
    const double *reciprocal_lattice_vectors, const int *mesh, const int *atm,
    const int *bas, const double *env, const int ranges[3]) {

  dim3 block_size(256);
  int a_range = ranges[0];
  int b_range = ranges[1];
  int c_range = ranges[2];
  const int mesh_a = mesh[0];
  const int mesh_b = mesh[1];
  const int mesh_c = mesh[2];

  dim3 block_grid(n_pairs / block_size.x + 1);
  switch (i_angular * 10 + j_angular) {
    diffused_density_kernel_case_macro(0, 0);
    diffused_density_kernel_case_macro(0, 1);
    diffused_density_kernel_case_macro(0, 2);
    diffused_density_kernel_case_macro(0, 3);
    diffused_density_kernel_case_macro(0, 4);
    diffused_density_kernel_case_macro(1, 0);
    diffused_density_kernel_case_macro(1, 1);
    diffused_density_kernel_case_macro(1, 2);
    diffused_density_kernel_case_macro(1, 3);
    diffused_density_kernel_case_macro(1, 4);
    diffused_density_kernel_case_macro(2, 0);
    diffused_density_kernel_case_macro(2, 1);
    diffused_density_kernel_case_macro(2, 2);
    diffused_density_kernel_case_macro(2, 3);
    diffused_density_kernel_case_macro(2, 4);
    diffused_density_kernel_case_macro(3, 0);
    diffused_density_kernel_case_macro(3, 1);
    diffused_density_kernel_case_macro(3, 2);
    diffused_density_kernel_case_macro(3, 3);
    diffused_density_kernel_case_macro(3, 4);
    diffused_density_kernel_case_macro(4, 0);
    diffused_density_kernel_case_macro(4, 1);
    diffused_density_kernel_case_macro(4, 2);
    diffused_density_kernel_case_macro(4, 3);
    diffused_density_kernel_case_macro(4, 4);
  default:
    fprintf(stderr,
            "angular momentum pair %d, %d is not supported in "
            "evaluate_density_driver\n",
            i_angular, j_angular);
  }

  checkCudaErrors(cudaPeekAtLastError());
}

template <int n_channels, int i_angular, int j_angular>
__global__ void evaluate_xc_kernel(
    double *fock, const double *xc_weights, const int *non_trivial_pairs,
    const double *cutoffs, const int *i_shells, const int *j_shells,
    const int n_j_shells, const int *shell_to_ao_indices,
    const int n_i_functions, const int n_j_functions,
    const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int mesh_a, const int mesh_b, const int mesh_c, const int *atm,
    const int *bas, const double *env) {
  const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
  const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
  const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
  const uint thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                         threadIdx.z * blockDim.y * blockDim.x;

  bool out_of_boundary = a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c;

  const uint flattened_index =
      a_index * mesh_b * mesh_c + b_index * mesh_c + c_index;
  const int xc_weights_stride = mesh_a * mesh_b * mesh_c;
  const int fock_stride = n_i_functions * n_j_functions;

  const double position_x = lattice_vectors[0] * a_index / mesh_a +
                            lattice_vectors[3] * b_index / mesh_b +
                            lattice_vectors[6] * c_index / mesh_c;
  const double position_y = lattice_vectors[1] * a_index / mesh_a +
                            lattice_vectors[4] * b_index / mesh_b +
                            lattice_vectors[7] * c_index / mesh_c;
  const double position_z = lattice_vectors[2] * a_index / mesh_a +
                            lattice_vectors[5] * b_index / mesh_b +
                            lattice_vectors[8] * c_index / mesh_c;

  constexpr int n_i_cartesian_functions = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_j_cartesian_functions = (j_angular + 1) * (j_angular + 2) / 2;
  double xc_values[n_channels];
  double neighboring_gaussian_sum[n_i_cartesian_functions *
                                  n_j_cartesian_functions];

  double i_cartesian[n_i_cartesian_functions];
  double j_cartesian[n_j_cartesian_functions];

  __shared__ double shared_memory[64];

#pragma unroll
  for (int i_channel = 0; i_channel < n_channels; i_channel++) {
    xc_values[i_channel] =
        xc_weights[i_channel * xc_weights_stride + flattened_index];
  }

  const uint local_grid_index =
      blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

  for (int i_pair_index = accumulated_n_pairs_per_local_grid[local_grid_index];
       i_pair_index < accumulated_n_pairs_per_local_grid[local_grid_index + 1];
       i_pair_index++) {

    const int i_pair = sorted_pairs_per_local_grid[i_pair_index];
    const int image_index = image_indices[i_pair];

    const double cutoff = cutoffs[i_pair];

    const double image_x = vectors_to_neighboring_images[image_index * 3];
    const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
    const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

    const int shell_pair_index = non_trivial_pairs[i_pair];
    const int i_shell_index = shell_pair_index / n_j_shells;
    const int j_shell_index = shell_pair_index % n_j_shells;
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
    const double ij_exponent_in_prefactor =
        i_exponent * j_exponent / ij_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);

    double pair_prefactor = 0;
    if (!out_of_boundary) {
      pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff *
                                    j_coeff * common_fac_sp<i_angular>() *
                                    common_fac_sp<j_angular>();
    }

    const double pair_x = (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
    const double pair_y = (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
    const double pair_z = (i_exponent * i_z + j_exponent * j_z) / ij_exponent;

    const int lower_a_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[0] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[1] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[2]);
    const int upper_a_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[0] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[1] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[2]);

    const int lower_b_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[3] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[4] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[5]);
    const int upper_b_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[3] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[4] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[5]);

    const int lower_c_index =
        ceil((position_x - pair_x - cutoff) * reciprocal_lattice_vectors[6] +
             (position_y - pair_y - cutoff) * reciprocal_lattice_vectors[7] +
             (position_z - pair_z - cutoff) * reciprocal_lattice_vectors[8]);
    const int upper_c_index =
        floor((position_x - pair_x + cutoff) * reciprocal_lattice_vectors[6] +
              (position_y - pair_y + cutoff) * reciprocal_lattice_vectors[7] +
              (position_z - pair_z + cutoff) * reciprocal_lattice_vectors[8]);

#pragma unroll
    for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
         i_function_index++) {
      for (int j_function_index = 0; j_function_index < n_j_cartesian_functions;
           j_function_index++) {
        neighboring_gaussian_sum[i_function_index * n_j_cartesian_functions +
                                 j_function_index] = 0;
      }
    }
    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++) {
      for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++) {
        for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++) {

          const double x = position_x - a_cell * lattice_vectors[0] -
                           b_cell * lattice_vectors[3] -
                           c_cell * lattice_vectors[6];
          const double y = position_y - a_cell * lattice_vectors[1] -
                           b_cell * lattice_vectors[4] -
                           c_cell * lattice_vectors[7];
          const double z = position_z - a_cell * lattice_vectors[2] -
                           b_cell * lattice_vectors[5] -
                           c_cell * lattice_vectors[8];
          gto_cartesian<i_angular>(i_cartesian, x - i_x, y - i_y, z - i_z);
          gto_cartesian<j_angular>(j_cartesian, x - j_x, y - j_y, z - j_z);

          const double r_squared =
              distance_squared(x - pair_x, y - pair_y, z - pair_z);
          const double gaussian = exp(-ij_exponent * r_squared);
#pragma unroll
          for (int i_function_index = 0;
               i_function_index < n_i_cartesian_functions; i_function_index++) {
            for (int j_function_index = 0;
                 j_function_index < n_j_cartesian_functions;
                 j_function_index++) {
              neighboring_gaussian_sum[i_function_index *
                                           n_j_cartesian_functions +
                                       j_function_index] +=
                  gaussian * i_cartesian[i_function_index] *
                  j_cartesian[j_function_index];
            }
          }
        }
      }
    }

    double *fock_pointer = fock + image_index * fock_stride +
                           i_function * n_j_functions + j_function;
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++) {
      for (int i_function_index = 0; i_function_index < n_i_cartesian_functions;
           i_function_index++) {
        for (int j_function_index = 0;
             j_function_index < n_j_cartesian_functions; j_function_index++) {
          reduce_and_add<double, 64>(
              shared_memory, fock_pointer, thread_id,
              xc_values[i_channel] * pair_prefactor *
                  neighboring_gaussian_sum[i_function_index *
                                               n_j_cartesian_functions +
                                           j_function_index]);
          fock_pointer++;
        }
        fock_pointer += n_j_functions - n_j_cartesian_functions;
      }
      fock_pointer +=
          n_images * fock_stride - n_i_cartesian_functions * n_j_functions;
    }
  }
}

#define xc_kernel_macro(li, lj)                                                \
  evaluate_xc_kernel<n_channels, li, lj><<<block_grid, block_size>>>(          \
      fock, xc_weights, non_trivial_pairs, cutoffs, i_shells, j_shells,        \
      n_j_shells, shell_to_ao_indices, n_i_functions, n_j_functions,           \
      sorted_pairs_per_local_grid, accumulated_n_pairs_per_local_grid,         \
      image_indices, vectors_to_neighboring_images, n_images, lattice_vectors, \
      reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env)

#define xc_kernel_case_macro(li, lj)                                           \
  case (li * 10 + lj):                                                         \
    xc_kernel_macro(li, lj);                                                   \
    break

template <int n_channels>
void evaluate_xc_driver(
    double *fock, const double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int *mesh, const int *atm, const int *bas, const double *env,
    const int blocking_sizes[3]) {
  dim3 block_size(blocking_sizes[2], blocking_sizes[1], blocking_sizes[0]);
  int mesh_a = mesh[0];
  int mesh_b = mesh[1];
  int mesh_c = mesh[2];
  dim3 block_grid((mesh_c + blocking_sizes[2] - 1) / blocking_sizes[2],
                  (mesh_b + blocking_sizes[1] - 1) / blocking_sizes[1],
                  (mesh_a + blocking_sizes[0] - 1) / blocking_sizes[0]);

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
    fprintf(
        stderr,
        "angular momentum pair %d, %d is not supported in evaluate_xc_driver\n",
        i_angular, j_angular);
  }

  checkCudaErrors(cudaPeekAtLastError());
}

extern "C" {
void evaluate_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int *mesh, const int *atm, const int *bas, const double *env,
    const int blocking_sizes[3], const int n_channels) {
  if (n_channels == 1) {
    evaluate_density_driver<1>(
        density, density_matrices, i_angular, j_angular, non_trivial_pairs,
        cutoffs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
        n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
        accumulated_n_pairs_per_local_grid, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
  } else if (n_channels == 2) {
    evaluate_density_driver<2>(
        density, density_matrices, i_angular, j_angular, non_trivial_pairs,
        cutoffs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
        n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
        accumulated_n_pairs_per_local_grid, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
  } else {
    fprintf(stderr, "n_channels more than 2 is not supported.\n");
  }
}

void evaluate_diffused_density_driver(
    double *density, const double *density_matrices, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const int n_pairs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *contributing_area_begin,
    const int *image_indices, const double *vectors_to_neighboring_images,
    const int n_images, const double *lattice_vectors,
    const double *reciprocal_lattice_vectors, const int *mesh, const int *atm,
    const int *bas, const double *env, const int ranges[3],
    const int n_channels) {
  if (n_channels == 1) {
    evaluate_density_driver<1>(
        density, density_matrices, i_angular, j_angular, non_trivial_pairs,
        n_pairs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
        n_i_functions, n_j_functions, contributing_area_begin, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, ranges);
  } else if (n_channels == 2) {
    evaluate_density_driver<2>(
        density, density_matrices, i_angular, j_angular, non_trivial_pairs,
        n_pairs, i_shells, j_shells, n_j_shells, shell_to_ao_indices,
        n_i_functions, n_j_functions, contributing_area_begin, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, ranges);
  } else {
    fprintf(stderr, "n_channels more than 2 is not supported.\n");
  }
}

void evaluate_xc_driver(
    double *fock, const double *xc_weights, const int i_angular,
    const int j_angular, const int *non_trivial_pairs, const double *cutoffs,
    const int *i_shells, const int *j_shells, const int n_j_shells,
    const int *shell_to_ao_indices, const int n_i_functions,
    const int n_j_functions, const int *sorted_pairs_per_local_grid,
    const int *accumulated_n_pairs_per_local_grid, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const double *lattice_vectors, const double *reciprocal_lattice_vectors,
    const int *mesh, const int *atm, const int *bas, const double *env,
    const int blocking_sizes[3], const int n_channels) {
  if (n_channels == 1) {
    evaluate_xc_driver<1>(
        fock, xc_weights, i_angular, j_angular, non_trivial_pairs, cutoffs,
        i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
        n_j_functions, sorted_pairs_per_local_grid,
        accumulated_n_pairs_per_local_grid, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
  } else if (n_channels == 2) {
    evaluate_xc_driver<2>(
        fock, xc_weights, i_angular, j_angular, non_trivial_pairs, cutoffs,
        i_shells, j_shells, n_j_shells, shell_to_ao_indices, n_i_functions,
        n_j_functions, sorted_pairs_per_local_grid,
        accumulated_n_pairs_per_local_grid, image_indices,
        vectors_to_neighboring_images, n_images, lattice_vectors,
        reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
  } else {
    fprintf(stderr, "n_channels more than 2 is not supported.\n");
  }
}
}
