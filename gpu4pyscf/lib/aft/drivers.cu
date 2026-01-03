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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cartesian.cuh"
#include "evaluation.cuh"

namespace gpu4pyscf::aft {

__constant__ double G[9];
__constant__ double G_norm[3];
__constant__ double gaussian_cutoff_table[99];

template <int i_angular, int j_angular>
__global__ void
check_cartesian(cuda::std::complex<double> *cartesian, const double *density,
                const double *G, const int n_grid, const double px,
                const double py, const double pz, const double qx,
                const double qy, const double qz, const double exponent) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_grid) {
    return;
  }

  double gx, gy, gz;
  gx = G[tid];
  gy = G[tid + n_grid];
  gz = G[tid + 2 * n_grid];

  auto value = contract_with_density<i_angular, j_angular>(
      density, exponent, gx, gy, gz, px, py, pz, qx, qy, qz);

  cartesian[tid] = value;
}
} // namespace gpu4pyscf::aft

extern "C" {
void update_reciprocal_lattice_vectors(
    const double *reciprocal_lattice_vectors_on_device,
    const double *reciprocal_norm_on_device) {
  cudaMemcpyToSymbol(gpu4pyscf::aft::G, reciprocal_lattice_vectors_on_device,
                     9 * sizeof(double));
  cudaMemcpyToSymbol(gpu4pyscf::aft::G_norm, reciprocal_norm_on_device,
                     3 * sizeof(double));
}

void check_cartesian(cuda::std::complex<double> *cartesian,
                     const double *density, const double *G, const int n_grid,
                     const double px, const double py, const double pz,
                     const double qx, const double qy, const double qz,
                     const double exponent, const int i_angular,
                     const int j_angular) {
  switch (i_angular * 10 + j_angular) {
  case 0:
    gpu4pyscf::aft::check_cartesian<0, 0><<<(n_grid + 255) / 256, 256>>>(
        cartesian, density, G, n_grid, px, py, pz, qx, qy, qz, exponent);
    break;
  case 10:
    gpu4pyscf::aft::check_cartesian<1, 0><<<(n_grid + 255) / 256, 256>>>(
        cartesian, density, G, n_grid, px, py, pz, qx, qy, qz, exponent);
    break;
  case 01:
    gpu4pyscf::aft::check_cartesian<0, 1><<<(n_grid + 255) / 256, 256>>>(
        cartesian, density, G, n_grid, px, py, pz, qx, qy, qz, exponent);
    break;
  case 11:
    gpu4pyscf::aft::check_cartesian<1, 1><<<(n_grid + 255) / 256, 256>>>(
        cartesian, density, G, n_grid, px, py, pz, qx, qy, qz, exponent);
    break;

  case 22:
    gpu4pyscf::aft::check_cartesian<2, 2><<<(n_grid + 255) / 256, 256>>>(
        cartesian, density, G, n_grid, px, py, pz, qx, qy, qz, exponent);
    break;
  }
}
}
