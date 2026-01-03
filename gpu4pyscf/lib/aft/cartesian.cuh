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

template <int angular>
__forceinline__ __device__ void
hermite_polynomial(cuda::std::complex<double> hermite[],
                   const double renormalized_g, const double sqrt_exponent) {

  const cuda::std::complex<double> factor{0.0, -sqrt_exponent};
  cuda::std::complex<double> accumulated_factor = factor;

  double hermite_real_0 = 1;
  double hermite_real_1 = 2.0 * renormalized_g;

  hermite[0] = 1.0;

  if constexpr (angular >= 1) {
    hermite[1] = factor * hermite_real_1;
  }

#pragma unroll
  for (int i = 0; i < angular - 1; i++) {
    const double hermite_real =
        hermite_real_1 * 2.0 * renormalized_g - 2.0 * (i + 1) * hermite_real_0;
    accumulated_factor = accumulated_factor * factor;
    hermite[i + 2] = hermite_real * accumulated_factor;
    hermite_real_0 = hermite_real_1;
    hermite_real_1 = hermite_real;
  }
}

template <int angular>
__forceinline__ __device__ void power_series(double array[], const double x) {
  array[0] = 1;
#pragma unroll
  for (int i = 1; i <= angular; i++) {
    array[i] = array[i - 1] * x;
  }
}

template <int i_angular, int j_angular>
__forceinline__ __device__ cuda::std::complex<double>
contract_with_density(const double density[], const double exponent,
                      const double gx, const double gy, const double gz,
                      const double px, const double py, const double pz,
                      const double qx, const double qy, const double qz) {

  constexpr int total_angular = i_angular + j_angular;
  cuda::std::complex<double> result = 0;
  const double sqrt_exponent = sqrt(exponent);

  cuda::std::complex<double> hx[total_angular + 1];
  hermite_polynomial<total_angular>(hx, sqrt_exponent * gx, sqrt_exponent);
  cuda::std::complex<double> hy[total_angular + 1];
  hermite_polynomial<total_angular>(hy, sqrt_exponent * gy, sqrt_exponent);
  cuda::std::complex<double> hz[total_angular + 1];
  hermite_polynomial<total_angular>(hz, sqrt_exponent * gz, sqrt_exponent);

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  double ax[i_angular + 1];
  power_series<i_angular>(ax, px);
  double ay[i_angular + 1];
  power_series<i_angular>(ay, py);
  double az[i_angular + 1];
  power_series<i_angular>(az, pz);

  double bx[j_angular + 1];
  power_series<j_angular>(bx, qx);
  double by[j_angular + 1];
  power_series<j_angular>(by, qy);
  double bz[j_angular + 1];
  power_series<j_angular>(bz, qz);

  if constexpr (i_angular == 0 && j_angular == 0) {
    result += density[0] * (1);
  }
  if constexpr (i_angular == 0 && j_angular == 1) {
    result += density[0] * (bx[1] + hx[1]);
    result += density[1] * (by[1] + hy[1]);
    result += density[2] * (bz[1] + hz[1]);
  }
  if constexpr (i_angular == 0 && j_angular == 2) {
    result += density[0] * (bx[2] + 2 * bx[1] * hx[1] + hx[2]);
    result += density[1] * ((bx[1] + hx[1]) * (by[1] + hy[1]));
    result += density[2] * ((bx[1] + hx[1]) * (bz[1] + hz[1]));
    result += density[3] * (by[2] + 2 * by[1] * hy[1] + hy[2]);
    result += density[4] * ((by[1] + hy[1]) * (bz[1] + hz[1]));
    result += density[5] * (bz[2] + 2 * bz[1] * hz[1] + hz[2]);
  }
  if constexpr (i_angular == 1 && j_angular == 0) {
    result += density[0] * (ax[1] + hx[1]);
    result += density[1] * (ay[1] + hy[1]);
    result += density[2] * (az[1] + hz[1]);
  }
  if constexpr (i_angular == 1 && j_angular == 1) {
    result += density[0] * (bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]);
    result += density[1] * ((ax[1] + hx[1]) * (by[1] + hy[1]));
    result += density[2] * ((ax[1] + hx[1]) * (bz[1] + hz[1]));
    result += density[3] * ((bx[1] + hx[1]) * (ay[1] + hy[1]));
    result += density[4] * (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]);
    result += density[5] * ((ay[1] + hy[1]) * (bz[1] + hz[1]));
    result += density[6] * ((bx[1] + hx[1]) * (az[1] + hz[1]));
    result += density[7] * ((by[1] + hy[1]) * (az[1] + hz[1]));
    result += density[8] * (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]);
  }
  if constexpr (i_angular == 1 && j_angular == 2) {
    result +=
        density[0] * (bx[2] * hx[1] + 2 * bx[1] * hx[2] +
                      ax[1] * (bx[2] + 2 * bx[1] * hx[1] + hx[2]) + hx[3]);
    result += density[1] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (by[1] + hy[1]));
    result += density[2] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (bz[1] + hz[1]));
    result +=
        density[3] * ((ax[1] + hx[1]) * (by[2] + 2 * by[1] * hy[1] + hy[2]));
    result +=
        density[4] * ((ax[1] + hx[1]) * (by[1] + hy[1]) * (bz[1] + hz[1]));
    result +=
        density[5] * ((ax[1] + hx[1]) * (bz[2] + 2 * bz[1] * hz[1] + hz[2]));
    result +=
        density[6] * ((bx[2] + 2 * bx[1] * hx[1] + hx[2]) * (ay[1] + hy[1]));
    result += density[7] * ((bx[1] + hx[1]) *
                            (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]));
    result +=
        density[8] * ((bx[1] + hx[1]) * (ay[1] + hy[1]) * (bz[1] + hz[1]));
    result +=
        density[9] * (by[2] * hy[1] + 2 * by[1] * hy[2] +
                      ay[1] * (by[2] + 2 * by[1] * hy[1] + hy[2]) + hy[3]);
    result += density[10] * ((by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]) *
                             (bz[1] + hz[1]));
    result +=
        density[11] * ((ay[1] + hy[1]) * (bz[2] + 2 * bz[1] * hz[1] + hz[2]));
    result +=
        density[12] * ((bx[2] + 2 * bx[1] * hx[1] + hx[2]) * (az[1] + hz[1]));
    result +=
        density[13] * ((bx[1] + hx[1]) * (by[1] + hy[1]) * (az[1] + hz[1]));
    result += density[14] * ((bx[1] + hx[1]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[15] * ((by[2] + 2 * by[1] * hy[1] + hy[2]) * (az[1] + hz[1]));
    result += density[16] * ((by[1] + hy[1]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[17] * (bz[2] * hz[1] + 2 * bz[1] * hz[2] +
                       az[1] * (bz[2] + 2 * bz[1] * hz[1] + hz[2]) + hz[3]);
  }
  if constexpr (i_angular == 2 && j_angular == 0) {
    result += density[0] * (ax[2] + 2 * ax[1] * hx[1] + hx[2]);
    result += density[1] * ((ax[1] + hx[1]) * (ay[1] + hy[1]));
    result += density[2] * ((ax[1] + hx[1]) * (az[1] + hz[1]));
    result += density[3] * (ay[2] + 2 * ay[1] * hy[1] + hy[2]);
    result += density[4] * ((ay[1] + hy[1]) * (az[1] + hz[1]));
    result += density[5] * (az[2] + 2 * az[1] * hz[1] + hz[2]);
  }
  if constexpr (i_angular == 2 && j_angular == 1) {
    result += density[0] * (ax[2] * (bx[1] + hx[1]) + bx[1] * hx[2] +
                            2 * ax[1] * (bx[1] * hx[1] + hx[2]) + hx[3]);
    result +=
        density[1] * ((ax[2] + 2 * ax[1] * hx[1] + hx[2]) * (by[1] + hy[1]));
    result +=
        density[2] * ((ax[2] + 2 * ax[1] * hx[1] + hx[2]) * (bz[1] + hz[1]));
    result += density[3] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (ay[1] + hy[1]));
    result += density[4] * ((ax[1] + hx[1]) *
                            (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]));
    result +=
        density[5] * ((ax[1] + hx[1]) * (ay[1] + hy[1]) * (bz[1] + hz[1]));
    result += density[6] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (az[1] + hz[1]));
    result +=
        density[7] * ((ax[1] + hx[1]) * (by[1] + hy[1]) * (az[1] + hz[1]));
    result += density[8] * ((ax[1] + hx[1]) *
                            (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[9] * ((bx[1] + hx[1]) * (ay[2] + 2 * ay[1] * hy[1] + hy[2]));
    result += density[10] * (ay[2] * (by[1] + hy[1]) + by[1] * hy[2] +
                             2 * ay[1] * (by[1] * hy[1] + hy[2]) + hy[3]);
    result +=
        density[11] * ((ay[2] + 2 * ay[1] * hy[1] + hy[2]) * (bz[1] + hz[1]));
    result +=
        density[12] * ((bx[1] + hx[1]) * (ay[1] + hy[1]) * (az[1] + hz[1]));
    result += density[13] * ((by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]) *
                             (az[1] + hz[1]));
    result += density[14] * ((ay[1] + hy[1]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[15] * ((bx[1] + hx[1]) * (az[2] + 2 * az[1] * hz[1] + hz[2]));
    result +=
        density[16] * ((by[1] + hy[1]) * (az[2] + 2 * az[1] * hz[1] + hz[2]));
    result += density[17] * (az[2] * (bz[1] + hz[1]) + bz[1] * hz[2] +
                             2 * az[1] * (bz[1] * hz[1] + hz[2]) + hz[3]);
  }
  if constexpr (i_angular == 2 && j_angular == 2) {
    result += density[0] *
              (bx[2] * hx[2] + ax[2] * (bx[2] + 2 * bx[1] * hx[1] + hx[2]) +
               2 * bx[1] * hx[3] +
               2 * ax[1] * (bx[2] * hx[1] + 2 * bx[1] * hx[2] + hx[3]) + hx[4]);
    result += density[1] * ((ax[2] * (bx[1] + hx[1]) + bx[1] * hx[2] +
                             2 * ax[1] * (bx[1] * hx[1] + hx[2]) + hx[3]) *
                            (by[1] + hy[1]));
    result += density[2] * ((ax[2] * (bx[1] + hx[1]) + bx[1] * hx[2] +
                             2 * ax[1] * (bx[1] * hx[1] + hx[2]) + hx[3]) *
                            (bz[1] + hz[1]));
    result += density[3] * ((ax[2] + 2 * ax[1] * hx[1] + hx[2]) *
                            (by[2] + 2 * by[1] * hy[1] + hy[2]));
    result += density[4] * ((ax[2] + 2 * ax[1] * hx[1] + hx[2]) *
                            (by[1] + hy[1]) * (bz[1] + hz[1]));
    result += density[5] * ((ax[2] + 2 * ax[1] * hx[1] + hx[2]) *
                            (bz[2] + 2 * bz[1] * hz[1] + hz[2]));
    result +=
        density[6] * ((bx[2] * hx[1] + 2 * bx[1] * hx[2] +
                       ax[1] * (bx[2] + 2 * bx[1] * hx[1] + hx[2]) + hx[3]) *
                      (ay[1] + hy[1]));
    result += density[7] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]));
    result += density[8] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                            (ay[1] + hy[1]) * (bz[1] + hz[1]));
    result +=
        density[9] * ((ax[1] + hx[1]) *
                      (by[2] * hy[1] + 2 * by[1] * hy[2] +
                       ay[1] * (by[2] + 2 * by[1] * hy[1] + hy[2]) + hy[3]));
    result += density[10] * ((ax[1] + hx[1]) *
                             (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]) *
                             (bz[1] + hz[1]));
    result += density[11] * ((ax[1] + hx[1]) * (ay[1] + hy[1]) *
                             (bz[2] + 2 * bz[1] * hz[1] + hz[2]));
    result +=
        density[12] * ((bx[2] * hx[1] + 2 * bx[1] * hx[2] +
                        ax[1] * (bx[2] + 2 * bx[1] * hx[1] + hx[2]) + hx[3]) *
                       (az[1] + hz[1]));
    result += density[13] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                             (by[1] + hy[1]) * (az[1] + hz[1]));
    result += density[14] * ((bx[1] * hx[1] + ax[1] * (bx[1] + hx[1]) + hx[2]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[15] * ((ax[1] + hx[1]) * (by[2] + 2 * by[1] * hy[1] + hy[2]) *
                       (az[1] + hz[1]));
    result += density[16] * ((ax[1] + hx[1]) * (by[1] + hy[1]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[17] * ((ax[1] + hx[1]) *
                       (bz[2] * hz[1] + 2 * bz[1] * hz[2] +
                        az[1] * (bz[2] + 2 * bz[1] * hz[1] + hz[2]) + hz[3]));
    result += density[18] * ((bx[2] + 2 * bx[1] * hx[1] + hx[2]) *
                             (ay[2] + 2 * ay[1] * hy[1] + hy[2]));
    result += density[19] *
              ((bx[1] + hx[1]) * (ay[2] * (by[1] + hy[1]) + by[1] * hy[2] +
                                  2 * ay[1] * (by[1] * hy[1] + hy[2]) + hy[3]));
    result +=
        density[20] * ((bx[1] + hx[1]) * (ay[2] + 2 * ay[1] * hy[1] + hy[2]) *
                       (bz[1] + hz[1]));
    result += density[21] *
              (by[2] * hy[2] + ay[2] * (by[2] + 2 * by[1] * hy[1] + hy[2]) +
               2 * by[1] * hy[3] +
               2 * ay[1] * (by[2] * hy[1] + 2 * by[1] * hy[2] + hy[3]) + hy[4]);
    result += density[22] * ((ay[2] * (by[1] + hy[1]) + by[1] * hy[2] +
                              2 * ay[1] * (by[1] * hy[1] + hy[2]) + hy[3]) *
                             (bz[1] + hz[1]));
    result += density[23] * ((ay[2] + 2 * ay[1] * hy[1] + hy[2]) *
                             (bz[2] + 2 * bz[1] * hz[1] + hz[2]));
    result += density[24] * ((bx[2] + 2 * bx[1] * hx[1] + hx[2]) *
                             (ay[1] + hy[1]) * (az[1] + hz[1]));
    result += density[25] * ((bx[1] + hx[1]) *
                             (by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]) *
                             (az[1] + hz[1]));
    result += density[26] * ((bx[1] + hx[1]) * (ay[1] + hy[1]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[27] * ((by[2] * hy[1] + 2 * by[1] * hy[2] +
                        ay[1] * (by[2] + 2 * by[1] * hy[1] + hy[2]) + hy[3]) *
                       (az[1] + hz[1]));
    result += density[28] * ((by[1] * hy[1] + ay[1] * (by[1] + hy[1]) + hy[2]) *
                             (bz[1] * hz[1] + az[1] * (bz[1] + hz[1]) + hz[2]));
    result +=
        density[29] * ((ay[1] + hy[1]) *
                       (bz[2] * hz[1] + 2 * bz[1] * hz[2] +
                        az[1] * (bz[2] + 2 * bz[1] * hz[1] + hz[2]) + hz[3]));
    result += density[30] * ((bx[2] + 2 * bx[1] * hx[1] + hx[2]) *
                             (az[2] + 2 * az[1] * hz[1] + hz[2]));
    result += density[31] * ((bx[1] + hx[1]) * (by[1] + hy[1]) *
                             (az[2] + 2 * az[1] * hz[1] + hz[2]));
    result += density[32] *
              ((bx[1] + hx[1]) * (az[2] * (bz[1] + hz[1]) + bz[1] * hz[2] +
                                  2 * az[1] * (bz[1] * hz[1] + hz[2]) + hz[3]));
    result += density[33] * ((by[2] + 2 * by[1] * hy[1] + hy[2]) *
                             (az[2] + 2 * az[1] * hz[1] + hz[2]));
    result += density[34] *
              ((by[1] + hy[1]) * (az[2] * (bz[1] + hz[1]) + bz[1] * hz[2] +
                                  2 * az[1] * (bz[1] * hz[1] + hz[2]) + hz[3]));
    result += density[35] *
              (bz[2] * hz[2] + az[2] * (bz[2] + 2 * bz[1] * hz[1] + hz[2]) +
               2 * bz[1] * hz[3] +
               2 * az[1] * (bz[2] * hz[1] + 2 * bz[1] * hz[2] + hz[3]) + hz[4]);
  }

  return result;
}

} // namespace gpu4pyscf::aft
