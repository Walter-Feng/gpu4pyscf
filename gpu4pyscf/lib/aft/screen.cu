/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>

#include "constant_objects.cuh"
#include "gint/gint.h"
#include "multigrid/multigrid_v2/utils.cuh"

namespace gpu4pyscf::aft {

// s[i, n] := Solve[ Integrate[4 Pi g^2 g^i Exp[- g^2], {g, g0, Infinity}] ==
// 10^(-n), g0]

__constant__ double gaussian_cutoff_table[99] = {
    1.56458, 2.24417, 2.74084, 3.15292, 3.51327, 3.83771,
    4.13534, 4.41192, 4.67142, 4.91669, 5.14986,

    1.81533, 2.46995, 2.95244, 3.35429, 3.70657, 4.02436,
    4.3163,  4.58795, 4.84308, 5.08443, 5.31406,

    2.09092, 2.70638, 3.17011, 3.55956, 3.90255, 4.2129,
    4.49865, 4.76499, 5.01549, 5.25274, 5.47867,

    2.37682, 2.94857, 3.39132, 3.76715, 4.10012, 4.40256,
    4.6818,  4.9426,  5.1883,  5.4213,  5.64344,

    2.66372, 3.19284, 3.61404, 3.97578, 4.2984,  4.59269,
    4.86524, 5.12038, 5.36117, 5.58986, 5.80815,

    2.9466,  3.4366,  3.83673, 4.18444, 4.49665, 4.78274,
    5.04855, 5.29798, 5.53383, 5.75818, 5.97261,

    3.22305, 3.67817, 4.0583,  4.39234, 4.69431, 4.97228,
    5.23138, 5.47513, 5.70606, 5.92608, 6.13664,

    3.49208, 3.91647, 4.27794, 4.5989,  4.89094, 5.16095,
    5.41346, 5.6516,  5.87766, 6.09339, 6.30012,

    3.7535,  4.1509,  4.49513, 4.8037,  5.08618, 5.34848,
    5.59455, 5.82719, 6.04847, 6.25997, 6.46293,
};

// s[n] := Integrate[4 Pi g^2 g^n Exp[- g^2], {g, 0, Infinity}]
__constant__ double gaussian_integration_values[9] = {
    5.56833, 6.28319, 8.35249, 12.5664, 20.8812,
    37.6991, 73.0843, 150.796, 328.879};

using gpu4pyscf::gpbc::multi_grid::distance_squared;

__global__ void
count_non_trivial_pairs_kernel(int *n_counts, const int n_primitives,
                               const double *vectors_to_neighboring_images,
                               const int n_images, const int *atm,
                               const int *bas, const double *env,
                               const int log_precision) {
  int i_shell = threadIdx.x + blockDim.x * blockIdx.x;
  int j_shell = threadIdx.y + blockDim.y * blockIdx.y;
  int image_index = threadIdx.z + blockDim.z * blockIdx.z;
  bool is_valid_pair = i_shell < n_primitives && j_shell < n_primitives &&
                       image_index < n_images;

  if (!is_valid_pair) {
    i_shell = 0;
    j_shell = 0;
    image_index = 0;
  }

  const double i_exponent = env[bas(PTR_EXP, i_shell)];
  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
  const double i_x =
      env[i_coord_offset] + vectors_to_neighboring_images[image_index * 3];
  const double i_y = env[i_coord_offset + 1] +
                     vectors_to_neighboring_images[image_index * 3 + 1];
  const double i_z = env[i_coord_offset + 2] +
                     vectors_to_neighboring_images[image_index * 3 + 2];

  const double i_coeff = env[bas(PTR_COEFF, i_shell)];
  const int i_angular = bas(ANG_OF, i_shell);

  const double j_exponent = env[bas(PTR_EXP, j_shell)];
  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
  const double j_x = env[j_coord_offset];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];
  const double j_coeff = env[bas(PTR_COEFF, j_shell)];
  const int j_angular = bas(ANG_OF, j_shell);

  const double pair_exponent = i_exponent + j_exponent;
  const double pair_exponent_in_prefactor =
      i_exponent * j_exponent / pair_exponent *
      distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);
  const double reciprocal_pair_exponent = 0.25 / pair_exponent;

  const int pair_angular = i_angular + j_angular;

  double fourier_factor = M_PI / pair_exponent;
  fourier_factor *= fourier_factor * fourier_factor;

  const double pair_prefactor =
      i_coeff * j_coeff * sqrt(fourier_factor) *
      pow(sqrt(reciprocal_pair_exponent), pair_angular + 2) *
      pow(2, pair_angular);

  const double estimated_integral_value =
      pair_prefactor * exp(-pair_exponent_in_prefactor) *
      gaussian_integration_values[pair_angular];

  if (estimated_integral_value < pow(10.0, -log_precision)) {
    is_valid_pair = false;
  }

  int count = is_valid_pair ? 1 : 0;
  count =
      cub::BlockReduce<int, 16, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, 16>()
          .Sum(count);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    atomicAdd(n_counts, count);
  }
}

__global__ void
screen_gaussian_pairs_kernel(int *shell_list, int *image_list, int *angulars,
                             double *cutoffs, int *written_counts,
                             const int n_primitives,
                             const double *vectors_to_neighboring_images,
                             const int n_images, const int *atm, const int *bas,
                             const double *env, const int log_precision) {
  int i_shell = threadIdx.x + blockDim.x * blockIdx.x;
  int j_shell = threadIdx.y + blockDim.y * blockIdx.y;
  int image_index = threadIdx.z + blockDim.z * blockIdx.z;
  bool is_valid_pair = i_shell < n_primitives && j_shell < n_primitives &&
                       image_index < n_images;

  if (!is_valid_pair) {
    i_shell = 0;
    j_shell = 0;
    image_index = 0;
  }

  const double i_exponent = env[bas(PTR_EXP, i_shell)];
  const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
  const double i_x =
      env[i_coord_offset] + vectors_to_neighboring_images[image_index * 3];
  const double i_y = env[i_coord_offset + 1] +
                     vectors_to_neighboring_images[image_index * 3 + 1];
  const double i_z = env[i_coord_offset + 2] +
                     vectors_to_neighboring_images[image_index * 3 + 2];

  const double i_coeff = env[bas(PTR_COEFF, i_shell)];
  const int i_angular = bas(ANG_OF, i_shell);

  const double j_exponent = env[bas(PTR_EXP, j_shell)];
  const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
  const double j_x = env[j_coord_offset];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];
  const double j_coeff = env[bas(PTR_COEFF, j_shell)];
  const int j_angular = bas(ANG_OF, j_shell);

  const double pair_exponent = i_exponent + j_exponent;
  const double pair_exponent_in_prefactor =
      i_exponent * j_exponent / pair_exponent *
      distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);
  const double reciprocal_pair_exponent = 0.25 / pair_exponent;

  const int pair_angular = i_angular + j_angular;

  double fourier_factor = M_PI / pair_exponent;
  fourier_factor *= fourier_factor * fourier_factor;

  const double pair_prefactor =
      i_coeff * j_coeff * sqrt(fourier_factor) *
      pow(sqrt(reciprocal_pair_exponent), pair_angular + 2) *
      pow(2, pair_angular);

  const double estimated_integral_value =
      pair_prefactor * exp(-pair_exponent_in_prefactor) *
      gaussian_integration_values[pair_angular];

  if (estimated_integral_value < pow(10.0, -log_precision)) {
    is_valid_pair = false;
  }

  int write_pair_index = is_valid_pair ? 1 : 0;
  int aggregated_pairs;
  cub::BlockScan<int, 16, cub::BLOCK_SCAN_RAKING, 16>().ExclusiveSum(
      write_pair_index, write_pair_index, aggregated_pairs);
  __shared__ int write_offset_for_this_block;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    write_offset_for_this_block = atomicAdd(written_counts, aggregated_pairs);
  }
  __syncthreads();

  const int write_offset_for_this_thread =
      write_offset_for_this_block + write_pair_index;

  if (is_valid_pair) {
    int equivalent_offset_for_cutoff = ceil(log10(estimated_integral_value));
    int offset = log_precision + equivalent_offset_for_cutoff;
    if (offset < 0)
      offset = 0;
    if (offset > 10)
      offset = 10;

    const double cutoff = gaussian_cutoff_table[pair_angular * 11 + offset] /
                          sqrt(reciprocal_pair_exponent);
    shell_list[write_offset_for_this_thread] = i_shell * n_primitives + j_shell;
    image_list[write_offset_for_this_thread] = image_index;
    angulars[write_offset_for_this_thread] = i_angular * 10 + j_angular;
    cutoffs[write_offset_for_this_thread] = cutoff;
  }
}

__global__ void count_pairs_on_blocks_kernel(
    int *counts_on_blocks, const double *cutoffs, const int n_pairs,
    const int mesh_a, const int mesh_b, const int mesh_c, const int n_a_blocks,
    const int n_b_blocks, const int n_c_blocks) {

  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;
  const int thread_id =
      threadIdx.x + BLOCK_DIM_XYZ * (threadIdx.y + BLOCK_DIM_XYZ * threadIdx.z);

  const int block_a_index = blockIdx.x;
  const int block_b_index = blockIdx.y;
  const int block_c_index = blockIdx.z;

  const bool reverse_a = block_a_index >= n_a_blocks / 2;
  const bool reverse_b = block_b_index >= n_b_blocks / 2;
  const bool reverse_c = block_c_index >= n_c_blocks / 2;

  const int a_begin_index =
      reverse_a ? (block_a_index - n_a_blocks + 1) * BLOCK_DIM_XYZ - 1
                : block_a_index * BLOCK_DIM_XYZ;
  const int b_begin_index =
      reverse_b ? (block_b_index - n_b_blocks + 1) * BLOCK_DIM_XYZ - 1
                : block_b_index * BLOCK_DIM_XYZ;
  const int c_begin_index =
      reverse_c ? (block_c_index - n_c_blocks + 1) * BLOCK_DIM_XYZ - 1
                : block_c_index * BLOCK_DIM_XYZ;

  const double gx_begin =
      G[0] * a_begin_index + G[3] * b_begin_index + G[6] * c_begin_index;
  const double gy_begin =
      G[1] * a_begin_index + G[4] * b_begin_index + G[7] * c_begin_index;
  const double gz_begin =
      G[2] * a_begin_index + G[5] * b_begin_index + G[8] * c_begin_index;

  const double distance = sqrt(distance_squared(gx_begin, gy_begin, gz_begin));

  int count = 0;
  const int n_batches = n_pairs / n_threads + 1;
  for (int i = thread_id, i_batch = 0; i_batch < n_batches;
       i_batch++, i += n_threads) {
    if (i < n_pairs) {
      const double cutoff = cutoffs[i];
      if (distance > cutoff) {
        break;
      } else {
        count++;
      }
    }
  }

  count = cub::BlockReduce<int, BLOCK_DIM_XYZ,
                           cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                           BLOCK_DIM_XYZ, BLOCK_DIM_XYZ>()
              .Sum(count);

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    counts_on_blocks[block_a_index * n_b_blocks * n_c_blocks +
                     block_b_index * n_c_blocks + block_c_index] = count;
    if (count > 0) {
      atomicAdd(counts_on_blocks + n_a_blocks * n_b_blocks * n_c_blocks, 1);
    }
  }
}
} // namespace gpu4pyscf::aft

extern "C" {

void count_non_trivial_pairs(int *n_counts, const int n_primitives,
                             const double *vectors_to_neighboring_images,
                             const int n_images, const int *atm, const int *bas,
                             const double *env, const int log_precision) {

  const dim3 block_size{16, 16, 1};
  const dim3 block_grid{(uint)ceil((double)n_primitives / block_size.x),
                        (uint)ceil((double)n_primitives / block_size.y),
                        (uint)ceil((double)n_images / block_size.z)};

  gpu4pyscf::aft::count_non_trivial_pairs_kernel<<<block_grid, block_size>>>(
      n_counts, n_primitives, vectors_to_neighboring_images, n_images, atm, bas,
      env, log_precision);
}

void screen_gaussian_pairs(int *shell_list, int *image_list, int *angulars,
                           double *cutoffs, int *written_counts,
                           const int n_primitives,
                           const double *vectors_to_neighboring_images,
                           const int n_images, const int *atm, const int *bas,
                           const double *env, const int log_precision) {

  const dim3 block_size{16, 16, 1};
  const dim3 block_grid{(uint)ceil((double)n_primitives / block_size.x),
                        (uint)ceil((double)n_primitives / block_size.y),
                        (uint)ceil((double)n_images / block_size.z)};

  gpu4pyscf::aft::screen_gaussian_pairs_kernel<<<block_grid, block_size>>>(
      shell_list, image_list, angulars, cutoffs, written_counts, n_primitives,
      vectors_to_neighboring_images, n_images, atm, bas, env, log_precision);
}

void count_pairs_on_blocks(int *counts_on_blocks, const double *cutoffs,
                           const int n_pairs, const int mesh_a,
                           const int mesh_b, const int mesh_c,
                           const int n_a_blocks, const int n_b_blocks,
                           const int n_c_blocks) {

  const dim3 block_size{BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ};
  const dim3 block_grid{(uint)n_a_blocks, (uint)n_b_blocks, (uint)n_c_blocks};

  gpu4pyscf::aft::count_pairs_on_blocks_kernel<<<block_grid, block_size>>>(
      counts_on_blocks, cutoffs, n_pairs, mesh_a, mesh_b, mesh_c, n_a_blocks,
      n_b_blocks, n_c_blocks);
}
}
