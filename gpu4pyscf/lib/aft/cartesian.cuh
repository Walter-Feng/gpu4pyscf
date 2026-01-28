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

// It is assumed that hermite[1] stores I * 2.0 * g
template <typename T, int angular>
__forceinline__ __device__ void hermite_polynomial(complex<T> hermite[],
                                                   const T exponent) {
#pragma unroll
  for (int i = 0; i < angular - 1; i++) {
    hermite[i + 2] =
        hermite[1] * hermite[i + 1] + 2.0 * exponent * (i + 1) * hermite[i];
  }
}

template <typename T, int i_angular, int j_angular>
__forceinline__ __device__ void
horizontal_recursion(T result[], const double shift_to_here) {
  if constexpr (i_angular == 1 && j_angular == 0) {
    result[1] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 1) {
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 2) {
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 3) {
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 4) {
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 2 && j_angular == 0) {
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[2] = result[2] + shift_to_here * result[1];
  }
  if constexpr (i_angular == 2 && j_angular == 1) {
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
  }
  if constexpr (i_angular == 2 && j_angular == 2) {
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
  }
  if constexpr (i_angular == 2 && j_angular == 3) {
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
  }
  if constexpr (i_angular == 2 && j_angular == 4) {
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
  }
  if constexpr (i_angular == 3 && j_angular == 0) {
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[3] = result[3] + shift_to_here * result[2];
  }
  if constexpr (i_angular == 3 && j_angular == 1) {
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
  }
  if constexpr (i_angular == 3 && j_angular == 2) {
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
  }
  if constexpr (i_angular == 3 && j_angular == 3) {
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
  }
  if constexpr (i_angular == 3 && j_angular == 4) {
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
  }
  if constexpr (i_angular == 4 && j_angular == 0) {
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[4] = result[4] + shift_to_here * result[3];
  }
  if constexpr (i_angular == 4 && j_angular == 1) {
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
  }
  if constexpr (i_angular == 4 && j_angular == 2) {
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
  }
  if constexpr (i_angular == 4 && j_angular == 3) {
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[16] = result[13] + shift_to_here * result[12];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[19] = result[16] + shift_to_here * result[15];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
  }
  if constexpr (i_angular == 4 && j_angular == 4) {
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[20] = result[16] + shift_to_here * result[15];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[24] = result[20] + shift_to_here * result[19];
    result[23] = result[19] + shift_to_here * result[18];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
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
    result += density[0] * (+xij[0] * yij[0] * zij[0]);
  }
  if constexpr (ai == 0 && aj == 1) {
    result += density[0] * (+xij[1] * yij[0] * zij[0]);
    result += density[1] * (+xij[0] * yij[1] * zij[0]);
    result += density[2] * (+xij[0] * yij[0] * zij[1]);
  }
  if constexpr (ai == 0 && aj == 2) {
    result += density[0] * (+1.0925484305920792 * xij[1] * yij[1] * zij[0]);
    result += density[1] * (+1.0925484305920792 * xij[0] * yij[1] * zij[1]);
    result += density[2] * (-0.31539156525252 * xij[2] * yij[0] * zij[0] -
                            0.31539156525252 * xij[0] * yij[2] * zij[0] +
                            0.63078313050504 * xij[0] * yij[0] * zij[2]);
    result += density[3] * (+1.0925484305920792 * xij[1] * yij[0] * zij[1]);
    result += density[4] * (+0.5462742152960396 * xij[2] * yij[0] * zij[0] -
                            0.5462742152960396 * xij[0] * yij[2] * zij[0]);
  }
  if constexpr (ai == 1 && aj == 0) {
    result += density[0] * (+xij[1] * yij[0] * zij[0]);
    result += density[1] * (+xij[0] * yij[1] * zij[0]);
    result += density[2] * (+xij[0] * yij[0] * zij[1]);
  }
  if constexpr (ai == 1 && aj == 1) {
    result += density[0] * (+xij[3] * yij[0] * zij[0]);
    result += density[1] * (+xij[2] * yij[1] * zij[0]);
    result += density[2] * (+xij[2] * yij[0] * zij[1]);
    result += density[3] * (+xij[1] * yij[2] * zij[0]);
    result += density[4] * (+xij[0] * yij[3] * zij[0]);
    result += density[5] * (+xij[0] * yij[2] * zij[1]);
    result += density[6] * (+xij[1] * yij[0] * zij[2]);
    result += density[7] * (+xij[0] * yij[1] * zij[2]);
    result += density[8] * (+xij[0] * yij[0] * zij[3]);
  }
  if constexpr (ai == 1 && aj == 2) {
    result += density[0] * (+1.0925484305920792 * xij[4] * yij[1] * zij[0]);
    result += density[1] * (+1.0925484305920792 * xij[3] * yij[1] * zij[1]);
    result += density[2] * (-0.31539156525252 * xij[5] * yij[0] * zij[0] -
                            0.31539156525252 * xij[3] * yij[2] * zij[0] +
                            0.63078313050504 * xij[3] * yij[0] * zij[2]);
    result += density[3] * (+1.0925484305920792 * xij[4] * yij[0] * zij[1]);
    result += density[4] * (+0.5462742152960396 * xij[5] * yij[0] * zij[0] -
                            0.5462742152960396 * xij[3] * yij[2] * zij[0]);
    result += density[5] * (+1.0925484305920792 * xij[1] * yij[4] * zij[0]);
    result += density[6] * (+1.0925484305920792 * xij[0] * yij[4] * zij[1]);
    result += density[7] * (-0.31539156525252 * xij[2] * yij[3] * zij[0] -
                            0.31539156525252 * xij[0] * yij[5] * zij[0] +
                            0.63078313050504 * xij[0] * yij[3] * zij[2]);
    result += density[8] * (+1.0925484305920792 * xij[1] * yij[3] * zij[1]);
    result += density[9] * (+0.5462742152960396 * xij[2] * yij[3] * zij[0] -
                            0.5462742152960396 * xij[0] * yij[5] * zij[0]);
    result += density[10] * (+1.0925484305920792 * xij[1] * yij[1] * zij[3]);
    result += density[11] * (+1.0925484305920792 * xij[0] * yij[1] * zij[4]);
    result += density[12] * (-0.31539156525252 * xij[2] * yij[0] * zij[3] -
                             0.31539156525252 * xij[0] * yij[2] * zij[3] +
                             0.63078313050504 * xij[0] * yij[0] * zij[5]);
    result += density[13] * (+1.0925484305920792 * xij[1] * yij[0] * zij[4]);
    result += density[14] * (+0.5462742152960396 * xij[2] * yij[0] * zij[3] -
                             0.5462742152960396 * xij[0] * yij[2] * zij[3]);
  }
  if constexpr (ai == 2 && aj == 0) {
    result += density[0] * (+1.0925484305920792 * xij[1] * yij[1] * zij[0]);
    result += density[1] * (+1.0925484305920792 * xij[0] * yij[1] * zij[1]);
    result += density[2] * (-0.31539156525252 * xij[2] * yij[0] * zij[0] -
                            0.31539156525252 * xij[0] * yij[2] * zij[0] +
                            0.63078313050504 * xij[0] * yij[0] * zij[2]);
    result += density[3] * (+1.0925484305920792 * xij[1] * yij[0] * zij[1]);
    result += density[4] * (+0.5462742152960396 * xij[2] * yij[0] * zij[0] -
                            0.5462742152960396 * xij[0] * yij[2] * zij[0]);
  }
  if constexpr (ai == 2 && aj == 1) {
    result += density[0] * (+1.0925484305920792 * xij[3] * yij[2] * zij[0]);
    result += density[1] * (+1.0925484305920792 * xij[2] * yij[3] * zij[0]);
    result += density[2] * (+1.0925484305920792 * xij[2] * yij[2] * zij[1]);
    result += density[3] * (+1.0925484305920792 * xij[1] * yij[2] * zij[2]);
    result += density[4] * (+1.0925484305920792 * xij[0] * yij[3] * zij[2]);
    result += density[5] * (+1.0925484305920792 * xij[0] * yij[2] * zij[3]);
    result += density[6] * (-0.31539156525252 * xij[5] * yij[0] * zij[0] -
                            0.31539156525252 * xij[1] * yij[4] * zij[0] +
                            0.63078313050504 * xij[1] * yij[0] * zij[4]);
    result += density[7] * (-0.31539156525252 * xij[4] * yij[1] * zij[0] -
                            0.31539156525252 * xij[0] * yij[5] * zij[0] +
                            0.63078313050504 * xij[0] * yij[1] * zij[4]);
    result += density[8] * (-0.31539156525252 * xij[4] * yij[0] * zij[1] -
                            0.31539156525252 * xij[0] * yij[4] * zij[1] +
                            0.63078313050504 * xij[0] * yij[0] * zij[5]);
    result += density[9] * (+1.0925484305920792 * xij[3] * yij[0] * zij[2]);
    result += density[10] * (+1.0925484305920792 * xij[2] * yij[1] * zij[2]);
    result += density[11] * (+1.0925484305920792 * xij[2] * yij[0] * zij[3]);
    result += density[12] * (+0.5462742152960396 * xij[5] * yij[0] * zij[0] -
                             0.5462742152960396 * xij[1] * yij[4] * zij[0]);
    result += density[13] * (+0.5462742152960396 * xij[4] * yij[1] * zij[0] -
                             0.5462742152960396 * xij[0] * yij[5] * zij[0]);
    result += density[14] * (+0.5462742152960396 * xij[4] * yij[0] * zij[1] -
                             0.5462742152960396 * xij[0] * yij[4] * zij[1]);
  }
  if constexpr (ai == 2 && aj == 2) {
    result += density[0] * (+1.1936620731892154 * xij[4] * yij[4] * zij[0]);
    result += density[1] * (+1.1936620731892154 * xij[3] * yij[4] * zij[1]);
    result += density[2] * (-0.34458055963862005 * xij[5] * yij[3] * zij[0] -
                            0.34458055963862005 * xij[3] * yij[5] * zij[0] +
                            0.6891611192772401 * xij[3] * yij[3] * zij[2]);
    result += density[3] * (+1.1936620731892154 * xij[4] * yij[3] * zij[1]);
    result += density[4] * (+0.5968310365946077 * xij[5] * yij[3] * zij[0] -
                            0.5968310365946077 * xij[3] * yij[5] * zij[0]);
    result += density[5] * (+1.1936620731892154 * xij[1] * yij[4] * zij[3]);
    result += density[6] * (+1.1936620731892154 * xij[0] * yij[4] * zij[4]);
    result += density[7] * (-0.34458055963862005 * xij[2] * yij[3] * zij[3] -
                            0.34458055963862005 * xij[0] * yij[5] * zij[3] +
                            0.6891611192772401 * xij[0] * yij[3] * zij[5]);
    result += density[8] * (+1.1936620731892154 * xij[1] * yij[3] * zij[4]);
    result += density[9] * (+0.5968310365946077 * xij[2] * yij[3] * zij[3] -
                            0.5968310365946077 * xij[0] * yij[5] * zij[3]);
    result += density[10] * (-0.34458055963862005 * xij[7] * yij[1] * zij[0] -
                             0.34458055963862005 * xij[1] * yij[7] * zij[0] +
                             0.6891611192772401 * xij[1] * yij[1] * zij[6]);
    result += density[11] * (-0.34458055963862005 * xij[6] * yij[1] * zij[1] -
                             0.34458055963862005 * xij[0] * yij[7] * zij[1] +
                             0.6891611192772401 * xij[0] * yij[1] * zij[7]);
    result += density[12] * (+0.09947183943243458 * xij[8] * yij[0] * zij[0] +
                             0.09947183943243458 * xij[6] * yij[2] * zij[0] -
                             0.19894367886486916 * xij[6] * yij[0] * zij[2] +
                             0.09947183943243458 * xij[2] * yij[6] * zij[0] +
                             0.09947183943243458 * xij[0] * yij[8] * zij[0] -
                             0.19894367886486916 * xij[0] * yij[6] * zij[2] -
                             0.19894367886486916 * xij[2] * yij[0] * zij[6] -
                             0.19894367886486916 * xij[0] * yij[2] * zij[6] +
                             0.3978873577297383 * xij[0] * yij[0] * zij[8]);
    result += density[13] * (-0.34458055963862005 * xij[7] * yij[0] * zij[1] -
                             0.34458055963862005 * xij[1] * yij[6] * zij[1] +
                             0.6891611192772401 * xij[1] * yij[0] * zij[7]);
    result += density[14] * (-0.17229027981931003 * xij[8] * yij[0] * zij[0] +
                             0.17229027981931003 * xij[6] * yij[2] * zij[0] -
                             0.17229027981931003 * xij[2] * yij[6] * zij[0] +
                             0.17229027981931003 * xij[0] * yij[8] * zij[0] +
                             0.34458055963862005 * xij[2] * yij[0] * zij[6] -
                             0.34458055963862005 * xij[0] * yij[2] * zij[6]);
    result += density[15] * (+1.1936620731892154 * xij[4] * yij[1] * zij[3]);
    result += density[16] * (+1.1936620731892154 * xij[3] * yij[1] * zij[4]);
    result += density[17] * (-0.34458055963862005 * xij[5] * yij[0] * zij[3] -
                             0.34458055963862005 * xij[3] * yij[2] * zij[3] +
                             0.6891611192772401 * xij[3] * yij[0] * zij[5]);
    result += density[18] * (+1.1936620731892154 * xij[4] * yij[0] * zij[4]);
    result += density[19] * (+0.5968310365946077 * xij[5] * yij[0] * zij[3] -
                             0.5968310365946077 * xij[3] * yij[2] * zij[3]);
    result += density[20] * (+0.5968310365946077 * xij[7] * yij[1] * zij[0] -
                             0.5968310365946077 * xij[1] * yij[7] * zij[0]);
    result += density[21] * (+0.5968310365946077 * xij[6] * yij[1] * zij[1] -
                             0.5968310365946077 * xij[0] * yij[7] * zij[1]);
    result += density[22] * (-0.17229027981931003 * xij[8] * yij[0] * zij[0] -
                             0.17229027981931003 * xij[6] * yij[2] * zij[0] +
                             0.34458055963862005 * xij[6] * yij[0] * zij[2] +
                             0.17229027981931003 * xij[2] * yij[6] * zij[0] +
                             0.17229027981931003 * xij[0] * yij[8] * zij[0] -
                             0.34458055963862005 * xij[0] * yij[6] * zij[2]);
    result += density[23] * (+0.5968310365946077 * xij[7] * yij[0] * zij[1] -
                             0.5968310365946077 * xij[1] * yij[6] * zij[1]);
    result += density[24] * (+0.29841551829730384 * xij[8] * yij[0] * zij[0] -
                             0.29841551829730384 * xij[6] * yij[2] * zij[0] -
                             0.29841551829730384 * xij[2] * yij[6] * zij[0] +
                             0.29841551829730384 * xij[0] * yij[8] * zij[0]);
  }
  return result;
}

} // namespace gpu4pyscf::aft
