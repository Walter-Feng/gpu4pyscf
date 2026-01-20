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

#include <cuda/std/complex>

namespace gpu4pyscf::aft {
template <typename T> using complex = cuda::std::complex<T>;

template <typename T, int angular>
__forceinline__ __device__ void
hermite_polynomial(complex<T> hermite[], const complex<T> g1, const T exponent,
                   const T shift = 0.0) {

  hermite[0] = 1.0;

  if constexpr (angular >= 1) {
    hermite[1] = g1;
  }

#pragma unroll
  for (int i = -1; i < angular - 1; i++) {
    hermite[i + 2] =
        g1 * hermite[i + 1] + 2.0 * exponent * (i + 1) * hermite[i];
  }
}

template <typename T, int i_angular, int j_angular>
__forceinline__ __device__ void horizontal_recursion(complex<T> table[],
                                                     const T shift) {
  if constexpr (i_angular == 0 && j_angular == 1) {
    table[1] = table[1] + shift * table[0];
  }

  if constexpr (i_angular == 1 && j_angular == 1) {
    table[3] = table[2] + shift * table[1];
    table[2] = table[1] + shift * table[0];
  }

  if constexpr (i_angular == 2 && j_angular == 1) {
    table[5] = table[3] + shift * table[2];
    table[4] = table[2] + shift * table[1];
    table[3] = table[1] + shift * table[0];
  }

  if constexpr (i_angular == 0 && j_angular == 2) {
    table[2] = table[2] + shift * table[1];
    table[1] = table[1] + shift * table[0];
    table[2] = table[2] + shift * table[1];
  }

  if constexpr (i_angular == 1 && j_angular == 2) {
    table[4] = table[3] + shift * table[2];
    table[3] = table[2] + shift * table[1];
    table[2] = table[1] + shift * table[0];
    table[5] = table[4] + shift * table[3];
    table[4] = table[3] + shift * table[2];
  }

  if constexpr (i_angular == 2 && j_angular == 2) {
    table[6] = table[4] + shift * table[3];
    table[5] = table[3] + shift * table[2];
    table[4] = table[2] + shift * table[1];
    table[3] = table[1] + shift * table[0];
    table[8] = table[6] + shift * table[5];
    table[7] = table[5] + shift * table[4];
    table[6] = table[4] + shift * table[3];
  }
}

template <typename T, int length>
__forceinline__ __device__ void conjugate(complex<T> table[]) {
#pragma unroll
  for (int i = 0; i < length; i++) {
    table[i].imag() *= -1;
  }
}

template <typename T, int ai, int aj>
__forceinline__ __device__ complex<T>
contract_with_density(const T density[], const complex<T> xij[],
                      const complex<T> yij[], const complex<T> zij[]) {

  constexpr int total_angular = ai + aj;
  complex<T> result = 0;

  if constexpr (ai == 0 && aj == 0) {
    result += density[0] * 1;
  }

  if constexpr (ai == 0 && aj == 1) {
    result += density[0] * xij[1];
    result += density[1] * yij[1];
    result += density[2] * zij[1];
  }

  if constexpr (ai == 0 && aj == 2) {
    result += density[0] * xij[1] * xij[1];
    result += density[1] * xij[1] * yij[1];
    result += density[2] * xij[1] * zij[1];
    result += density[3] * yij[1] * yij[1];
    result += density[4] * yij[1] * zij[1];
    result += density[5] * zij[1] * zij[1];
  }

  if constexpr (ai == 1 && aj == 0) {
    result += density[0] * xij[aj + 1];
    result += density[1] * yij[aj + 1];
    result += density[2] * zij[aj + 1];
  }

  if constexpr (ai == 1 && aj == 1) {
    result += density[0] * xij[(aj + 1) + 1];
    result += density[1] * xij[aj + 1] * yij[1];
    result += density[2] * xij[aj + 1] * zij[1];
    result += density[3] * xij[1] * yij[aj + 1];
    result += density[4] * yij[(aj + 1) + 1];
    result += density[5] * yij[aj + 1] * zij[1];
    result += density[6] * xij[1] * zij[aj + 1];
    result += density[7] * yij[1] * zij[aj + 1];
    result += density[8] * zij[(aj + 1) + 1];
  }

  if constexpr (ai == 1 && aj == 2) {
    result += density[0] * xij[(aj + 1) * 2 + 1];
    result += density[1] * xij[(aj + 1) + 1] * yij[1];
    result += density[2] * xij[(aj + 1) + 1] * zij[1];
    result += density[3] * xij[aj + 1] * (yij[1] * yij[1]);
    result += density[4] * xij[aj + 1] * yij[1] * zij[1];
    result += density[5] * xij[aj + 1] * (zij[1] * zij[1]);
    result += density[6] * xij[1] * xij[1] * yij[aj + 1];
    result += density[7] * xij[1] * yij[(aj + 1) + 1];
    result += density[8] * xij[1] * yij[aj + 1] * zij[1];
    result += density[9] * yij[(aj + 1) * 2 + 1];
    result += density[10] * yij[(aj + 1) + 1] * zij[1];
    result += density[11] * yij[aj + 1] * (zij[1] * zij[1]);
    result += density[12] * xij[1] * xij[1] * zij[aj + 1];
    result += density[13] * xij[1] * yij[1] * zij[aj + 1];
    result += density[14] * xij[1] * zij[(aj + 1) + 1];
    result += density[15] * yij[1] * yij[1] * zij[aj + 1];
    result += density[16] * yij[1] * zij[(aj + 1) + 1];
    result += density[17] * zij[(aj + 1) * 2 + 1];
  }

  if constexpr (ai == 2 && aj == 0) {
    result += density[0] * xij[aj + 1] * xij[aj + 1];
    result += density[1] * xij[aj + 1] * yij[aj + 1];
    result += density[2] * xij[aj + 1] * zij[aj + 1];
    result += density[3] * yij[aj + 1] * yij[aj + 1];
    result += density[4] * yij[aj + 1] * zij[aj + 1];
    result += density[5] * zij[aj + 1] * zij[aj + 1];
  }

  if constexpr (ai == 2 && aj == 1) {
    result += density[0] * xij[(aj + 1) * 1 + 2];
    result += density[1] * xij[aj + 1] * xij[aj + 1] * yij[1];
    result += density[2] * xij[aj + 1] * xij[aj + 1] * zij[1];
    result += density[3] * xij[(aj + 1) + 1] * yij[aj + 1];
    result += density[4] * xij[aj + 1] * yij[(aj + 1) + 1];
    result += density[5] * xij[aj + 1] * yij[aj + 1] * zij[1];
    result += density[6] * xij[(aj + 1) + 1] * zij[aj + 1];
    result += density[7] * xij[aj + 1] * yij[1] * zij[aj + 1];
    result += density[8] * xij[aj + 1] * zij[(aj + 1) + 1];
    result += density[9] * xij[1] * (yij[aj + 1] * yij[aj + 1]);
    result += density[10] * yij[(aj + 1) * 1 + 2];
    result += density[11] * yij[aj + 1] * yij[aj + 1] * zij[1];
    result += density[12] * xij[1] * yij[aj + 1] * zij[aj + 1];
    result += density[13] * yij[(aj + 1) + 1] * zij[aj + 1];
    result += density[14] * yij[aj + 1] * zij[(aj + 1) + 1];
    result += density[15] * xij[1] * (zij[aj + 1] * zij[aj + 1]);
    result += density[16] * yij[1] * (zij[aj + 1] * zij[aj + 1]);
    result += density[17] * zij[(aj + 1) * 1 + 2];
  }

  if constexpr (ai == 2 && aj == 2) {
    result += density[0] * xij[aj * 2 + 2];
    result += density[1] * xij[(aj + 1) * 1 + 2] * yij[1];
    result += density[2] * xij[(aj + 1) * 1 + 2] * zij[1];
    result += density[3] * xij[aj + 1] * xij[aj + 1] * (yij[1] * yij[1]);
    result += density[4] * xij[aj + 1] * xij[aj + 1] * yij[1] * zij[1];
    result += density[5] * xij[aj + 1] * xij[aj + 1] * (zij[1] * zij[1]);
    result += density[6] * xij[(aj + 1) * 2 + 1] * yij[aj + 1];
    result += density[7] * xij[(aj + 1) + 1] * yij[(aj + 1) + 1];
    result += density[8] * xij[(aj + 1) + 1] * yij[aj + 1] * zij[1];
    result += density[9] * xij[aj + 1] * yij[(aj + 1) * 2 + 1];
    result += density[10] * xij[aj + 1] * yij[(aj + 1) + 1] * zij[1];
    result += density[11] * xij[aj + 1] * yij[aj + 1] * (zij[1] * zij[1]);
    result += density[12] * xij[(aj + 1) * 2 + 1] * zij[aj + 1];
    result += density[13] * xij[(aj + 1) + 1] * yij[1] * zij[aj + 1];
    result += density[14] * xij[(aj + 1) + 1] * zij[(aj + 1) + 1];
    result += density[15] * xij[aj + 1] * (yij[1] * yij[1]) * zij[aj + 1];
    result += density[16] * xij[aj + 1] * yij[1] * zij[(aj + 1) + 1];
    result += density[17] * xij[aj + 1] * zij[(aj + 1) * 2 + 1];
    result += density[18] * xij[1] * xij[1] * (yij[aj + 1] * yij[aj + 1]);
    result += density[19] * xij[1] * yij[(aj + 1) * 1 + 2];
    result += density[20] * xij[1] * (yij[aj + 1] * yij[aj + 1]) * zij[1];
    result += density[21] * yij[aj * 2 + 2];
    result += density[22] * yij[(aj + 1) * 1 + 2] * zij[1];
    result += density[23] * yij[aj + 1] * yij[aj + 1] * (zij[1] * zij[1]);
    result += density[24] * xij[1] * xij[1] * yij[aj + 1] * zij[aj + 1];
    result += density[25] * xij[1] * yij[(aj + 1) + 1] * zij[aj + 1];
    result += density[26] * xij[1] * yij[aj + 1] * zij[(aj + 1) + 1];
    result += density[27] * yij[(aj + 1) * 2 + 1] * zij[aj + 1];
    result += density[28] * yij[(aj + 1) + 1] * zij[(aj + 1) + 1];
    result += density[29] * yij[aj + 1] * zij[(aj + 1) * 2 + 1];
    result += density[30] * xij[1] * xij[1] * (zij[aj + 1] * zij[aj + 1]);
    result += density[31] * xij[1] * yij[1] * (zij[aj + 1] * zij[aj + 1]);
    result += density[32] * xij[1] * zij[(aj + 1) * 1 + 2];
    result += density[33] * yij[1] * yij[1] * (zij[aj + 1] * zij[aj + 1]);
    result += density[34] * yij[1] * zij[(aj + 1) * 1 + 2];
    result += density[35] * zij[aj * 2 + 2];
  }

  return result;
}

} // namespace gpu4pyscf::aft
