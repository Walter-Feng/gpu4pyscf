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

namespace gpu4pyscf::gpbc::aft {

template <int angular>
__forceinline__ __device__ void hermite_polynomial(double hermite[],
                                                   const double g) {

  hermite[0] = 1.0;
  if constexpr (angular >= 1) {
    hermite[0] = 2.0 * g;
  }

#pragma unroll
  for (int i = 0; i < angular - 1; i++) {
    hermite[i + 2] = 2.0 * g * hermite[i + 1] - 2.0 * (i + 1) * hermite[i];
  }
}

template <typename T, int i_angular, int j_angular>
__forceinline__ __device__ double
contract_with_density(const T density[], const double i_to_center[],
                      const double j_to_center[], const double exponent,
                      const double gx, const double gy, const double gz) {

  constexpr int total_angular = i_angular + j_angular;
  T result = 0;

  double hermite_x[total_angular + 1];
  hermite_polynomial<total_angular>(hermite_x, gx);
  double hermite_y[total_angular + 1];
  hermite_polynomial<total_angular>(hermite_y, gy);
  double hermite_z[total_angular + 1];
  hermite_polynomial<total_angular>(hermite_z, gz);

  if constexpr (i_angular == 1 && j_angular == 0) {
    result += density[0] * hermite_x[0] * hermite_y[0] * hermite_z[0];
  } else if constexpr (total_angular == 1) {
    result += density[0] * hermite_x[1] * hermite_y[0] * hermite_z[0];
    result += density[1] * hermite_x[0] * hermite_y[1] * hermite_z[0];
    result += density[2] * hermite_x[0] * hermite_y[0] * hermite_z[1];
  } else if constexpr (total_angular == 2) {
  }
  return result;
}

} // namespace gpu4pyscf::gpbc::aft
