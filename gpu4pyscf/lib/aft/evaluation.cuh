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

#pragma once

#include "cartesian.cuh"
#include "constant_objects.cuh"
#include "multigrid/multigrid_v2/cartesian.cuh"
#include "utils.cuh"
#include <assert.h>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cuda/std/complex>
#include <gint/cuda_alloc.cuh>
#include <gint/gint.h>
#include <stdio.h>

#define BLOCK_DIM_XYZ 4

namespace gpu4pyscf::aft {

template <typename KernelType, int i_angular, int j_angular,
          bool is_non_orthogonal>
__global__ static void evaluate_density_kernel(
    cuDoubleComplex *density, const KernelType *density_matrices,
    const int *non_trivial_pairs, const int *n_contributing_pairs_in_blocks,
    const int n_shells, const int *shell_to_ao_indices, const int n_functions,
    const int *sorted_block_index, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int n_a_blocks,
    const int n_b_blocks, const int n_c_blocks, const int *atm, const int *bas,
    const double *env) {

  constexpr int n_fi = (i_angular + 1) * (i_angular + 2) / 2;
  constexpr int n_fj = (j_angular + 1) * (j_angular + 2) / 2;
  constexpr int n_threads = BLOCK_DIM_XYZ * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  const int density_matrix_stride = n_functions * n_functions;

  const int block_index = sorted_block_index[blockIdx.x];

  const int block_a_stride = n_b_blocks * n_c_blocks;
  const int block_a_index = block_index / block_a_stride;
  const int block_ab_index = block_index % block_a_stride;
  const int block_b_index = block_ab_index / n_c_blocks;
  const int block_c_index = block_ab_index % n_a_blocks;

  const bool reverse_a = block_a_index > n_a_blocks / 2;
  const bool reverse_b = block_b_index > n_b_blocks / 2;
  const bool reverse_c = block_c_index > n_c_blocks / 2;

  const int a_begin =
      reverse_a ? block_a_index * BLOCK_DIM_XYZ
                : mesh_a - 1 + (block_a_index - n_a_blocks + 1) * BLOCK_DIM_XYZ;
  const int b_begin =
      reverse_b ? block_b_index * BLOCK_DIM_XYZ
                : mesh_b - 1 + (block_b_index - n_b_blocks + 1) * BLOCK_DIM_XYZ;
  const int c_begin =
      reverse_c ? block_c_index * BLOCK_DIM_XYZ
                : mesh_c - 1 + (block_c_index - n_c_blocks + 1) * BLOCK_DIM_XYZ;

  const KernelType gx_begin = G[0] * a_begin + G[3] * b_begin + G[6] * c_begin;
  const KernelType gy_begin = G[1] * a_begin + G[4] * b_begin + G[7] * c_begin;
  const KernelType gz_begin = G[2] * a_begin + G[5] * b_begin + G[8] * c_begin;

  const int8_t a_sign = reverse_a ? -1 : 1;
  const int8_t b_sign = reverse_b ? -1 : 1;
  const int8_t c_sign = reverse_c ? -1 : 1;

  const KernelType a_dot_b =
      (a_sign * b_sign) * (G[0] * G[3] + G[1] * G[4] + G[2] * G[5]);
  const KernelType a_dot_c =
      (a_sign * c_sign) * (G[0] * G[6] + G[1] * G[7] + G[2] * G[8]);
  const KernelType b_dot_c =
      (b_sign * c_sign) * (G[3] * G[6] + G[4] * G[7] + G[5] * G[8]);

  const int a_upper = a_begin + BLOCK_DIM_XYZ;
  const int b_upper = b_begin + BLOCK_DIM_XYZ;
  const int c_upper = c_begin + BLOCK_DIM_XYZ;

  const int thread_id = threadIdx.x + threadIdx.y * BLOCK_DIM_XYZ +
                        threadIdx.z * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ;

  KernelType prefactor[n_fi * n_fj];

  __shared__ KernelType reduced_density_values[n_threads];
  reduced_density_values[thread_id] = 0;

  const int n_pairs = n_contributing_pairs_in_blocks[block_index];

  const int n_batches = (n_pairs + n_threads - 1) / n_threads;

  for (int i_batch = 0, pair = thread_id; i_batch < n_batches;
       i_batch++, pair += n_threads) {
    const bool is_valid_pair = pair < n_pairs;
    const int shell_pair = is_valid_pair ? non_trivial_pairs[pair] : 0;
    const int i_shell = shell_pair / n_shells;
    const int j_shell = shell_pair % n_shells;
    const int i_function = shell_to_ao_indices[i_shell];
    const int j_function = shell_to_ao_indices[j_shell];

    const int image_index = image_indices[pair];

    const KernelType i_exponent = env[bas(PTR_EXP, i_shell)];
    const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
    const KernelType i_x =
        env[i_coord_offset] + vectors_to_neighboring_images[image_index * 3];
    const KernelType i_y = env[i_coord_offset + 1] +
                           vectors_to_neighboring_images[image_index * 3 + 1];
    const KernelType i_z = env[i_coord_offset + 2] +
                           vectors_to_neighboring_images[image_index * 3 + 2];
    const KernelType i_coeff = env[bas(PTR_COEFF, i_shell)];

    const KernelType j_exponent = env[bas(PTR_EXP, j_shell)];
    const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
    const KernelType j_x = env[j_coord_offset];
    const KernelType j_y = env[j_coord_offset + 1];
    const KernelType j_z = env[j_coord_offset + 2];
    const KernelType j_coeff = env[bas(PTR_COEFF, j_shell)];

    const KernelType pair_exponent = i_exponent + j_exponent;
    const KernelType pair_exponent_in_prefactor =
        i_exponent * j_exponent / pair_exponent *
        distance_squared(i_x - j_x, i_y - j_y, i_z - j_z);
    const KernelType reciprocal_pair_exponent = 0.25 / pair_exponent;

    const KernelType pair_x =
        (i_exponent * i_x + j_exponent * j_x) / pair_exponent;
    const KernelType pair_y =
        (i_exponent * i_y + j_exponent * j_y) / pair_exponent;
    const KernelType pair_z =
        (i_exponent * i_z + j_exponent * j_z) / pair_exponent;

    const KernelType px = pair_x - i_x;
    const KernelType py = pair_y - i_y;
    const KernelType pz = pair_z - i_z;

    const KernelType qx = pair_x - j_x;
    const KernelType qy = pair_y - j_y;
    const KernelType qz = pair_z - j_z;

    const KernelType gaussian_exponent_at_reference =
        reciprocal_pair_exponent *
        distance_squared(gx_begin, gy_begin, gz_begin);

    const KernelType pair_prefactor =
        i_coeff * j_coeff *
        gpbc::multi_grid::common_fac_sp<KernelType, i_angular>() *
        gpbc::multi_grid::common_fac_sp<KernelType, j_angular>();

    const KernelType gaussian_begin =
        is_valid_pair ? exp(-(pair_exponent_in_prefactor +
                              gaussian_exponent_at_reference) /
                            3.0)
                      : 0;
    const KernelType *density_matrix_pointer =
        density_matrices + image_index * density_matrix_stride +
        i_function * n_functions + j_function;

#pragma unroll
    for (int f_i = 0; f_i < n_fi; f_i++) {
#pragma unroll
      for (int f_j = 0; f_j < n_fj; f_j++) {
        const KernelType density_matrix_value = prefactor[f_i * n_fj + f_j] =
            pair_prefactor * density_matrix_pointer[f_j];
      }
    }

    const KernelType da_squared =
        gpbc::multi_grid::distance_squared(G[0], G[1], G[2]);
    const KernelType db_squared =
        gpbc::multi_grid::distance_squared(G[3], G[4], G[5]);
    const KernelType dc_squared =
        gpbc::multi_grid::distance_squared(G[6], G[7], G[8]);

    const KernelType exp_da_squared =
        exp(-2 * reciprocal_pair_exponent * da_squared);
    const KernelType exp_db_squared =
        exp(-2 * reciprocal_pair_exponent * db_squared);
    const KernelType exp_dc_squared =
        exp(-2 * reciprocal_pair_exponent * dc_squared);

    const KernelType cross_term_a =
        a_sign * (G[0] * gx_begin + G[1] * gy_begin + G[2] * gz_begin);
    const KernelType cross_term_b =
        b_sign * (G[3] * gx_begin + G[4] * gy_begin + G[5] * gz_begin);
    const KernelType cross_term_c =
        c_sign * (G[6] * gx_begin + G[7] * gy_begin + G[8] * gz_begin);

    const KernelType recursion_factor_a_begin =
        exp(-reciprocal_pair_exponent * (2 * cross_term_a + da_squared));
    const KernelType recursion_factor_b_begin =
        exp(-reciprocal_pair_exponent * (2 * cross_term_b + db_squared));
    const KernelType recursion_factor_c_begin =
        exp(-reciprocal_pair_exponent * (2 * cross_term_c + dc_squared));

    const KernelType exp_dadb = exp(-2 * reciprocal_pair_exponent * a_dot_b);
    const KernelType exp_dadc = exp(-2 * reciprocal_pair_exponent * a_dot_c);
    const KernelType exp_dbdc = exp(-2 * reciprocal_pair_exponent * b_dot_c);

    int a_index, b_index, c_index;
    KernelType gx, gy, gz;
    KernelType gaussian_x, gaussian_y, gaussian_z, recursion_factor_a,
        recursion_factor_b, recursion_factor_c;
    KernelType recursion_factor_ab_pow_a = 1;
    KernelType recursion_factor_ac_pow_a = 1;
    KernelType recursion_factor_bc_pow_b = 1;

    if constexpr (!is_non_orthogonal) {
      gx = gx_begin;
    }
    for (a_index = 0, gaussian_x = gaussian_begin,
        recursion_factor_a = recursion_factor_a_begin;
         a_index < a_upper; a_index++, gaussian_x *= recursion_factor_a,
        recursion_factor_a *= exp_da_squared) {
      if constexpr (is_non_orthogonal) {
        recursion_factor_bc_pow_b = 1;
      } else {
        gy = gy_begin;
      }
      for (b_index = 0, gaussian_y = gaussian_begin,
          recursion_factor_b = recursion_factor_b_begin;
           b_index < b_upper; b_index++,
          gaussian_y *= recursion_factor_b * recursion_factor_ab_pow_a,
          recursion_factor_b *= exp_db_squared) {

        if constexpr (is_non_orthogonal) {
          gx = gx_begin + a_index * a_sign * G[0] + b_index * b_sign * G[3];
          gy = gy_begin + a_index * a_sign * G[1] + b_index * b_sign * G[4];
          gz = gz_begin + a_index * a_sign * G[2] + b_index * b_sign * G[5];
        } else {
          gz = gz_begin;
        }
        for (c_index = 0, gaussian_z = gaussian_begin,
            recursion_factor_c = recursion_factor_c_begin;
             c_index < c_upper; c_index++,
            gaussian_z *= recursion_factor_c * recursion_factor_ac_pow_a *
                                recursion_factor_bc_pow_b,
            recursion_factor_c *= exp_dc_squared) {

          const KernelType gaussian = gaussian_x * gaussian_y * gaussian_z;
          cuDoubleComplex density_value_to_be_shared =
              contract_with_density<KernelType, i_angular, j_angular>(
                  prefactor, reciprocal_pair_exponent, gx, gy, gz, px, py, pz,
                  qx, qy, qz);

          density_value_to_be_shared *= gaussian;

          __syncthreads();

          const KernelType reduced =
              cub::BlockReduce<cuDoubleComplex, BLOCK_DIM_XYZ,
                               cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
                               BLOCK_DIM_XYZ, BLOCK_DIM_XYZ>()
                  .Sum(density_value_to_be_shared);

          if (thread_id == 0) {
            reduced_density_values[a_index * BLOCK_DIM_XYZ * BLOCK_DIM_XYZ +
                                   b_index * BLOCK_DIM_XYZ + c_index] +=
                reduced;
          }

          if constexpr (is_non_orthogonal) {
            gx += c_sign * G[6];
            gy += c_sign * G[7];
            gz += c_sign * G[8];
          } else {
            gz += c_sign * G[8];
          }
        }

        if constexpr (is_non_orthogonal) {
          recursion_factor_bc_pow_b *= exp_dbdc;
        } else {
          gy += b_sign * G[4];
        }
      }

      if constexpr (is_non_orthogonal) {
        recursion_factor_ab_pow_a *= exp_dadb;
        recursion_factor_ac_pow_a *= exp_dadc;
      } else {
        gx += a_sign * G[0];
      }
    }
  }
  const int a_index = a_begin + threadIdx.z;
  const int b_index = b_begin + threadIdx.y;
  const int c_index = c_begin + threadIdx.x;

  __syncthreads();

  if (a_index < mesh_a && b_index < mesh_b && c_index < mesh_c) {
    atomicAdd(density + a_index * mesh_b * mesh_c + b_index * mesh_c + c_index,
              reduced_density_values[thread_id]);
  }
}

} // namespace gpu4pyscf::aft
