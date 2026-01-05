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

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cartesian.cuh"
#include "evaluation.cuh"
#include "gint/cuda_alloc.cuh"

namespace gpu4pyscf::aft {

__constant__ double G[9];

} // namespace gpu4pyscf::aft

extern "C" {
void update_reciprocal_lattice_vectors(
    const double *reciprocal_lattice_vectors_on_device) {
  checkCudaErrors(cudaMemcpyToSymbol(gpu4pyscf::aft::G,
                                     reciprocal_lattice_vectors_on_device,
                                     9 * sizeof(double)));
}

int evaluate_density(
    cuda::std::complex<double> *density, const double *density_matrices,
    const int *non_trivial_pairs, const int n_shells,
    const int *n_contributing_pairs_in_blocks, const int *shell_to_ao_indices,
    const int n_functions, const int *sorted_block_index,
    const int n_contributing_blocks, const int *image_indices,
    const double *vectors_to_neighboring_images, const int n_images,
    const int mesh_a, const int mesh_b, const int mesh_c, const int n_a_blocks,
    const int n_b_blocks, const int n_c_blocks, const int *atm, const int *bas,
    const double *env, const int i_angular, const int j_angular) {

  const dim3 block_size{BLOCK_DIM_XYZ, BLOCK_DIM_XYZ, BLOCK_DIM_XYZ};
  const dim3 block_grid{(uint)n_contributing_blocks, 1, 1};
  switch (i_angular * 10 + j_angular) {
  case 0:
    gpu4pyscf::aft::evaluate_density_kernel<double, 0, 0, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 1:
    gpu4pyscf::aft::evaluate_density_kernel<double, 0, 1, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 10:
    gpu4pyscf::aft::evaluate_density_kernel<double, 1, 0, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 11:
    gpu4pyscf::aft::evaluate_density_kernel<double, 1, 1, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 2:
    gpu4pyscf::aft::evaluate_density_kernel<double, 0, 2, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 20:
    gpu4pyscf::aft::evaluate_density_kernel<double, 2, 0, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 12:
    gpu4pyscf::aft::evaluate_density_kernel<double, 1, 2, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 21:
    gpu4pyscf::aft::evaluate_density_kernel<double, 2, 1, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  case 22:
    gpu4pyscf::aft::evaluate_density_kernel<double, 2, 2, false>
        <<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, n_shells,
            n_contributing_pairs_in_blocks, shell_to_ao_indices, n_functions,
            sorted_block_index, image_indices, vectors_to_neighboring_images,
            n_images, mesh_a, mesh_b, mesh_c, n_a_blocks, n_b_blocks,
            n_c_blocks, atm, bas, env);
    break;
  }

  return checkCudaErrors(cudaPeekAtLastError());
}
}
