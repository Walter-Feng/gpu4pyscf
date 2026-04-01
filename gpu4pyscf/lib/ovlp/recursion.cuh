#pragma once

namespace rr {

template <int angular>
__forceinline__ __device__ constexpr double common_fac_sp() {
  if constexpr (angular == 0) {
    return 0.282094791773878143;
  } else if constexpr (angular == 1) {
    return 0.488602511902919921;
  } else {
    return 1.0;
  }
}

template <int angular>
__forceinline__ __device__ void
vertical_recursion(double result[], const double a00,
                   const double factor_from_previous,
                   const double factor_from_second_previous) {
  result[0] = a00;
  if constexpr (angular > 0) {
    result[1] = factor_from_previous * a00;
  }
#pragma unroll
  for (int i = 1; i < angular; i++) {
    result[i + 1] = i * factor_from_second_previous * result[i - 1] +
                    factor_from_previous * result[i];
  }
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void insert_position_operator(double result[],
                                                         const double shift) {
#pragma unroll
  for (int i = 0; i <= i_angular; i++) {
#pragma unroll
    for (int j = 0; j <= j_angular; j++) {
      result[i * stride + j] =
          result[i * stride + j + 1] + shift * result[i * stride + j];
    }
  }
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
insert_gradient_operator(double result[], const double recursion_factor) {
#pragma unroll
  for (int i = 0; i <= i_angular; i++) {
    double gradient, lower_order = 0;
#pragma unroll
    for (int j = 0; j <= j_angular; j++) {
      gradient =
          result[i * stride + j + 1] * recursion_factor - lower_order * j;
      lower_order = result[i * stride + j];
      result[i * stride + j] = gradient;
    }
  }
}

template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
insert_gradient_operator_to_bra(double result[],
                                const double recursion_factor) {
#pragma unroll
  for (int j = 0; j <= j_angular; j++) {
    double gradient, lower_order = 0;
#pragma unroll
    for (int i = 0; i <= i_angular; i++) {
      gradient =
          result[(i + 1) * stride + j] * recursion_factor - lower_order * i;
      lower_order = result[i * stride + j];
      result[i * stride + j] = gradient;
    }
  }
}

template <int i_angular, int j_angular>
__forceinline__ __device__ void
horizontal_recursion(double result[], const double shift_to_here) {
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
  if constexpr (i_angular == 1 && j_angular == 5) {
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 6) {
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
  }
  if constexpr (i_angular == 1 && j_angular == 7) {
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
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
  if constexpr (i_angular == 2 && j_angular == 5) {
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
  }
  if constexpr (i_angular == 2 && j_angular == 6) {
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
  }
  if constexpr (i_angular == 2 && j_angular == 7) {
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
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
  if constexpr (i_angular == 3 && j_angular == 5) {
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[18] = result[13] + shift_to_here * result[12];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[23] = result[18] + shift_to_here * result[17];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
  }
  if constexpr (i_angular == 3 && j_angular == 6) {
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[21] = result[15] + shift_to_here * result[14];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[27] = result[21] + shift_to_here * result[20];
    result[26] = result[20] + shift_to_here * result[19];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
  }
  if constexpr (i_angular == 3 && j_angular == 7) {
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[24] = result[17] + shift_to_here * result[16];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[31] = result[24] + shift_to_here * result[23];
    result[30] = result[23] + shift_to_here * result[22];
    result[29] = result[22] + shift_to_here * result[21];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
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
  if constexpr (i_angular == 4 && j_angular == 5) {
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[24] = result[19] + shift_to_here * result[18];
    result[23] = result[18] + shift_to_here * result[17];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[29] = result[24] + shift_to_here * result[23];
    result[28] = result[23] + shift_to_here * result[22];
    result[27] = result[22] + shift_to_here * result[21];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
  }
  if constexpr (i_angular == 4 && j_angular == 6) {
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[28] = result[22] + shift_to_here * result[21];
    result[27] = result[21] + shift_to_here * result[20];
    result[26] = result[20] + shift_to_here * result[19];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[34] = result[28] + shift_to_here * result[27];
    result[33] = result[27] + shift_to_here * result[26];
    result[32] = result[26] + shift_to_here * result[25];
    result[31] = result[25] + shift_to_here * result[24];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
  }
  if constexpr (i_angular == 4 && j_angular == 7) {
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[32] = result[25] + shift_to_here * result[24];
    result[31] = result[24] + shift_to_here * result[23];
    result[30] = result[23] + shift_to_here * result[22];
    result[29] = result[22] + shift_to_here * result[21];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[39] = result[32] + shift_to_here * result[31];
    result[38] = result[31] + shift_to_here * result[30];
    result[37] = result[30] + shift_to_here * result[29];
    result[36] = result[29] + shift_to_here * result[28];
    result[35] = result[28] + shift_to_here * result[27];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
  }
  if constexpr (i_angular == 5 && j_angular == 0) {
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[5] = result[5] + shift_to_here * result[4];
  }
  if constexpr (i_angular == 5 && j_angular == 1) {
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
  }
  if constexpr (i_angular == 5 && j_angular == 2) {
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[15] = result[13] + shift_to_here * result[12];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[17] = result[15] + shift_to_here * result[14];
    result[16] = result[14] + shift_to_here * result[13];
    result[15] = result[13] + shift_to_here * result[12];
  }
  if constexpr (i_angular == 5 && j_angular == 3) {
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[20] = result[17] + shift_to_here * result[16];
    result[19] = result[16] + shift_to_here * result[15];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[23] = result[20] + shift_to_here * result[19];
    result[22] = result[19] + shift_to_here * result[18];
    result[21] = result[18] + shift_to_here * result[17];
    result[20] = result[17] + shift_to_here * result[16];
  }
  if constexpr (i_angular == 5 && j_angular == 4) {
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[25] = result[21] + shift_to_here * result[20];
    result[24] = result[20] + shift_to_here * result[19];
    result[23] = result[19] + shift_to_here * result[18];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[29] = result[25] + shift_to_here * result[24];
    result[28] = result[24] + shift_to_here * result[23];
    result[27] = result[23] + shift_to_here * result[22];
    result[26] = result[22] + shift_to_here * result[21];
    result[25] = result[21] + shift_to_here * result[20];
  }
  if constexpr (i_angular == 5 && j_angular == 5) {
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[23] = result[18] + shift_to_here * result[17];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[30] = result[25] + shift_to_here * result[24];
    result[29] = result[24] + shift_to_here * result[23];
    result[28] = result[23] + shift_to_here * result[22];
    result[27] = result[22] + shift_to_here * result[21];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[35] = result[30] + shift_to_here * result[29];
    result[34] = result[29] + shift_to_here * result[28];
    result[33] = result[28] + shift_to_here * result[27];
    result[32] = result[27] + shift_to_here * result[26];
    result[31] = result[26] + shift_to_here * result[25];
    result[30] = result[25] + shift_to_here * result[24];
  }
  if constexpr (i_angular == 5 && j_angular == 6) {
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[27] = result[21] + shift_to_here * result[20];
    result[26] = result[20] + shift_to_here * result[19];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[35] = result[29] + shift_to_here * result[28];
    result[34] = result[28] + shift_to_here * result[27];
    result[33] = result[27] + shift_to_here * result[26];
    result[32] = result[26] + shift_to_here * result[25];
    result[31] = result[25] + shift_to_here * result[24];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[41] = result[35] + shift_to_here * result[34];
    result[40] = result[34] + shift_to_here * result[33];
    result[39] = result[33] + shift_to_here * result[32];
    result[38] = result[32] + shift_to_here * result[31];
    result[37] = result[31] + shift_to_here * result[30];
    result[36] = result[30] + shift_to_here * result[29];
    result[35] = result[29] + shift_to_here * result[28];
  }
  if constexpr (i_angular == 5 && j_angular == 7) {
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[31] = result[24] + shift_to_here * result[23];
    result[30] = result[23] + shift_to_here * result[22];
    result[29] = result[22] + shift_to_here * result[21];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[40] = result[33] + shift_to_here * result[32];
    result[39] = result[32] + shift_to_here * result[31];
    result[38] = result[31] + shift_to_here * result[30];
    result[37] = result[30] + shift_to_here * result[29];
    result[36] = result[29] + shift_to_here * result[28];
    result[35] = result[28] + shift_to_here * result[27];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[47] = result[40] + shift_to_here * result[39];
    result[46] = result[39] + shift_to_here * result[38];
    result[45] = result[38] + shift_to_here * result[37];
    result[44] = result[37] + shift_to_here * result[36];
    result[43] = result[36] + shift_to_here * result[35];
    result[42] = result[35] + shift_to_here * result[34];
    result[41] = result[34] + shift_to_here * result[33];
    result[40] = result[33] + shift_to_here * result[32];
  }
  if constexpr (i_angular == 6 && j_angular == 0) {
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[6] = result[6] + shift_to_here * result[5];
  }
  if constexpr (i_angular == 6 && j_angular == 1) {
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[12] = result[11] + shift_to_here * result[10];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
    result[13] = result[12] + shift_to_here * result[11];
    result[12] = result[11] + shift_to_here * result[10];
  }
  if constexpr (i_angular == 6 && j_angular == 2) {
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[16] = result[14] + shift_to_here * result[13];
    result[15] = result[13] + shift_to_here * result[12];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[18] = result[16] + shift_to_here * result[15];
    result[17] = result[15] + shift_to_here * result[14];
    result[16] = result[14] + shift_to_here * result[13];
    result[15] = result[13] + shift_to_here * result[12];
    result[20] = result[18] + shift_to_here * result[17];
    result[19] = result[17] + shift_to_here * result[16];
    result[18] = result[16] + shift_to_here * result[15];
  }
  if constexpr (i_angular == 6 && j_angular == 3) {
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[21] = result[18] + shift_to_here * result[17];
    result[20] = result[17] + shift_to_here * result[16];
    result[19] = result[16] + shift_to_here * result[15];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[24] = result[21] + shift_to_here * result[20];
    result[23] = result[20] + shift_to_here * result[19];
    result[22] = result[19] + shift_to_here * result[18];
    result[21] = result[18] + shift_to_here * result[17];
    result[20] = result[17] + shift_to_here * result[16];
    result[27] = result[24] + shift_to_here * result[23];
    result[26] = result[23] + shift_to_here * result[22];
    result[25] = result[22] + shift_to_here * result[21];
    result[24] = result[21] + shift_to_here * result[20];
  }
  if constexpr (i_angular == 6 && j_angular == 4) {
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[26] = result[22] + shift_to_here * result[21];
    result[25] = result[21] + shift_to_here * result[20];
    result[24] = result[20] + shift_to_here * result[19];
    result[23] = result[19] + shift_to_here * result[18];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[30] = result[26] + shift_to_here * result[25];
    result[29] = result[25] + shift_to_here * result[24];
    result[28] = result[24] + shift_to_here * result[23];
    result[27] = result[23] + shift_to_here * result[22];
    result[26] = result[22] + shift_to_here * result[21];
    result[25] = result[21] + shift_to_here * result[20];
    result[34] = result[30] + shift_to_here * result[29];
    result[33] = result[29] + shift_to_here * result[28];
    result[32] = result[28] + shift_to_here * result[27];
    result[31] = result[27] + shift_to_here * result[26];
    result[30] = result[26] + shift_to_here * result[25];
  }
  if constexpr (i_angular == 6 && j_angular == 5) {
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[23] = result[18] + shift_to_here * result[17];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[31] = result[26] + shift_to_here * result[25];
    result[30] = result[25] + shift_to_here * result[24];
    result[29] = result[24] + shift_to_here * result[23];
    result[28] = result[23] + shift_to_here * result[22];
    result[27] = result[22] + shift_to_here * result[21];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[36] = result[31] + shift_to_here * result[30];
    result[35] = result[30] + shift_to_here * result[29];
    result[34] = result[29] + shift_to_here * result[28];
    result[33] = result[28] + shift_to_here * result[27];
    result[32] = result[27] + shift_to_here * result[26];
    result[31] = result[26] + shift_to_here * result[25];
    result[30] = result[25] + shift_to_here * result[24];
    result[41] = result[36] + shift_to_here * result[35];
    result[40] = result[35] + shift_to_here * result[34];
    result[39] = result[34] + shift_to_here * result[33];
    result[38] = result[33] + shift_to_here * result[32];
    result[37] = result[32] + shift_to_here * result[31];
    result[36] = result[31] + shift_to_here * result[30];
  }
  if constexpr (i_angular == 6 && j_angular == 6) {
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[27] = result[21] + shift_to_here * result[20];
    result[26] = result[20] + shift_to_here * result[19];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[36] = result[30] + shift_to_here * result[29];
    result[35] = result[29] + shift_to_here * result[28];
    result[34] = result[28] + shift_to_here * result[27];
    result[33] = result[27] + shift_to_here * result[26];
    result[32] = result[26] + shift_to_here * result[25];
    result[31] = result[25] + shift_to_here * result[24];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[42] = result[36] + shift_to_here * result[35];
    result[41] = result[35] + shift_to_here * result[34];
    result[40] = result[34] + shift_to_here * result[33];
    result[39] = result[33] + shift_to_here * result[32];
    result[38] = result[32] + shift_to_here * result[31];
    result[37] = result[31] + shift_to_here * result[30];
    result[36] = result[30] + shift_to_here * result[29];
    result[35] = result[29] + shift_to_here * result[28];
    result[48] = result[42] + shift_to_here * result[41];
    result[47] = result[41] + shift_to_here * result[40];
    result[46] = result[40] + shift_to_here * result[39];
    result[45] = result[39] + shift_to_here * result[38];
    result[44] = result[38] + shift_to_here * result[37];
    result[43] = result[37] + shift_to_here * result[36];
    result[42] = result[36] + shift_to_here * result[35];
  }
  if constexpr (i_angular == 6 && j_angular == 7) {
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[31] = result[24] + shift_to_here * result[23];
    result[30] = result[23] + shift_to_here * result[22];
    result[29] = result[22] + shift_to_here * result[21];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[41] = result[34] + shift_to_here * result[33];
    result[40] = result[33] + shift_to_here * result[32];
    result[39] = result[32] + shift_to_here * result[31];
    result[38] = result[31] + shift_to_here * result[30];
    result[37] = result[30] + shift_to_here * result[29];
    result[36] = result[29] + shift_to_here * result[28];
    result[35] = result[28] + shift_to_here * result[27];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[48] = result[41] + shift_to_here * result[40];
    result[47] = result[40] + shift_to_here * result[39];
    result[46] = result[39] + shift_to_here * result[38];
    result[45] = result[38] + shift_to_here * result[37];
    result[44] = result[37] + shift_to_here * result[36];
    result[43] = result[36] + shift_to_here * result[35];
    result[42] = result[35] + shift_to_here * result[34];
    result[41] = result[34] + shift_to_here * result[33];
    result[40] = result[33] + shift_to_here * result[32];
    result[55] = result[48] + shift_to_here * result[47];
    result[54] = result[47] + shift_to_here * result[46];
    result[53] = result[46] + shift_to_here * result[45];
    result[52] = result[45] + shift_to_here * result[44];
    result[51] = result[44] + shift_to_here * result[43];
    result[50] = result[43] + shift_to_here * result[42];
    result[49] = result[42] + shift_to_here * result[41];
    result[48] = result[41] + shift_to_here * result[40];
  }
  if constexpr (i_angular == 7 && j_angular == 0) {
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[1] = result[1] + shift_to_here * result[0];
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[2] = result[2] + shift_to_here * result[1];
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[3] = result[3] + shift_to_here * result[2];
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[4] = result[4] + shift_to_here * result[3];
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[5] = result[5] + shift_to_here * result[4];
    result[7] = result[7] + shift_to_here * result[6];
    result[6] = result[6] + shift_to_here * result[5];
    result[7] = result[7] + shift_to_here * result[6];
  }
  if constexpr (i_angular == 7 && j_angular == 1) {
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[3] = result[2] + shift_to_here * result[1];
    result[2] = result[1] + shift_to_here * result[0];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[5] = result[4] + shift_to_here * result[3];
    result[4] = result[3] + shift_to_here * result[2];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[7] = result[6] + shift_to_here * result[5];
    result[6] = result[5] + shift_to_here * result[4];
    result[12] = result[11] + shift_to_here * result[10];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
    result[9] = result[8] + shift_to_here * result[7];
    result[8] = result[7] + shift_to_here * result[6];
    result[13] = result[12] + shift_to_here * result[11];
    result[12] = result[11] + shift_to_here * result[10];
    result[11] = result[10] + shift_to_here * result[9];
    result[10] = result[9] + shift_to_here * result[8];
    result[14] = result[13] + shift_to_here * result[12];
    result[13] = result[12] + shift_to_here * result[11];
    result[12] = result[11] + shift_to_here * result[10];
    result[15] = result[14] + shift_to_here * result[13];
    result[14] = result[13] + shift_to_here * result[12];
  }
  if constexpr (i_angular == 7 && j_angular == 2) {
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[5] = result[3] + shift_to_here * result[2];
    result[4] = result[2] + shift_to_here * result[1];
    result[3] = result[1] + shift_to_here * result[0];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[8] = result[6] + shift_to_here * result[5];
    result[7] = result[5] + shift_to_here * result[4];
    result[6] = result[4] + shift_to_here * result[3];
    result[15] = result[13] + shift_to_here * result[12];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[11] = result[9] + shift_to_here * result[8];
    result[10] = result[8] + shift_to_here * result[7];
    result[9] = result[7] + shift_to_here * result[6];
    result[17] = result[15] + shift_to_here * result[14];
    result[16] = result[14] + shift_to_here * result[13];
    result[15] = result[13] + shift_to_here * result[12];
    result[14] = result[12] + shift_to_here * result[11];
    result[13] = result[11] + shift_to_here * result[10];
    result[12] = result[10] + shift_to_here * result[9];
    result[19] = result[17] + shift_to_here * result[16];
    result[18] = result[16] + shift_to_here * result[15];
    result[17] = result[15] + shift_to_here * result[14];
    result[16] = result[14] + shift_to_here * result[13];
    result[15] = result[13] + shift_to_here * result[12];
    result[21] = result[19] + shift_to_here * result[18];
    result[20] = result[18] + shift_to_here * result[17];
    result[19] = result[17] + shift_to_here * result[16];
    result[18] = result[16] + shift_to_here * result[15];
    result[23] = result[21] + shift_to_here * result[20];
    result[22] = result[20] + shift_to_here * result[19];
    result[21] = result[19] + shift_to_here * result[18];
  }
  if constexpr (i_angular == 7 && j_angular == 3) {
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[7] = result[4] + shift_to_here * result[3];
    result[6] = result[3] + shift_to_here * result[2];
    result[5] = result[2] + shift_to_here * result[1];
    result[4] = result[1] + shift_to_here * result[0];
    result[16] = result[13] + shift_to_here * result[12];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[11] = result[8] + shift_to_here * result[7];
    result[10] = result[7] + shift_to_here * result[6];
    result[9] = result[6] + shift_to_here * result[5];
    result[8] = result[5] + shift_to_here * result[4];
    result[19] = result[16] + shift_to_here * result[15];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[15] = result[12] + shift_to_here * result[11];
    result[14] = result[11] + shift_to_here * result[10];
    result[13] = result[10] + shift_to_here * result[9];
    result[12] = result[9] + shift_to_here * result[8];
    result[22] = result[19] + shift_to_here * result[18];
    result[21] = result[18] + shift_to_here * result[17];
    result[20] = result[17] + shift_to_here * result[16];
    result[19] = result[16] + shift_to_here * result[15];
    result[18] = result[15] + shift_to_here * result[14];
    result[17] = result[14] + shift_to_here * result[13];
    result[16] = result[13] + shift_to_here * result[12];
    result[25] = result[22] + shift_to_here * result[21];
    result[24] = result[21] + shift_to_here * result[20];
    result[23] = result[20] + shift_to_here * result[19];
    result[22] = result[19] + shift_to_here * result[18];
    result[21] = result[18] + shift_to_here * result[17];
    result[20] = result[17] + shift_to_here * result[16];
    result[28] = result[25] + shift_to_here * result[24];
    result[27] = result[24] + shift_to_here * result[23];
    result[26] = result[23] + shift_to_here * result[22];
    result[25] = result[22] + shift_to_here * result[21];
    result[24] = result[21] + shift_to_here * result[20];
    result[31] = result[28] + shift_to_here * result[27];
    result[30] = result[27] + shift_to_here * result[26];
    result[29] = result[26] + shift_to_here * result[25];
    result[28] = result[25] + shift_to_here * result[24];
  }
  if constexpr (i_angular == 7 && j_angular == 4) {
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[9] = result[5] + shift_to_here * result[4];
    result[8] = result[4] + shift_to_here * result[3];
    result[7] = result[3] + shift_to_here * result[2];
    result[6] = result[2] + shift_to_here * result[1];
    result[5] = result[1] + shift_to_here * result[0];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[14] = result[10] + shift_to_here * result[9];
    result[13] = result[9] + shift_to_here * result[8];
    result[12] = result[8] + shift_to_here * result[7];
    result[11] = result[7] + shift_to_here * result[6];
    result[10] = result[6] + shift_to_here * result[5];
    result[23] = result[19] + shift_to_here * result[18];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[19] = result[15] + shift_to_here * result[14];
    result[18] = result[14] + shift_to_here * result[13];
    result[17] = result[13] + shift_to_here * result[12];
    result[16] = result[12] + shift_to_here * result[11];
    result[15] = result[11] + shift_to_here * result[10];
    result[27] = result[23] + shift_to_here * result[22];
    result[26] = result[22] + shift_to_here * result[21];
    result[25] = result[21] + shift_to_here * result[20];
    result[24] = result[20] + shift_to_here * result[19];
    result[23] = result[19] + shift_to_here * result[18];
    result[22] = result[18] + shift_to_here * result[17];
    result[21] = result[17] + shift_to_here * result[16];
    result[20] = result[16] + shift_to_here * result[15];
    result[31] = result[27] + shift_to_here * result[26];
    result[30] = result[26] + shift_to_here * result[25];
    result[29] = result[25] + shift_to_here * result[24];
    result[28] = result[24] + shift_to_here * result[23];
    result[27] = result[23] + shift_to_here * result[22];
    result[26] = result[22] + shift_to_here * result[21];
    result[25] = result[21] + shift_to_here * result[20];
    result[35] = result[31] + shift_to_here * result[30];
    result[34] = result[30] + shift_to_here * result[29];
    result[33] = result[29] + shift_to_here * result[28];
    result[32] = result[28] + shift_to_here * result[27];
    result[31] = result[27] + shift_to_here * result[26];
    result[30] = result[26] + shift_to_here * result[25];
    result[39] = result[35] + shift_to_here * result[34];
    result[38] = result[34] + shift_to_here * result[33];
    result[37] = result[33] + shift_to_here * result[32];
    result[36] = result[32] + shift_to_here * result[31];
    result[35] = result[31] + shift_to_here * result[30];
  }
  if constexpr (i_angular == 7 && j_angular == 5) {
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[11] = result[6] + shift_to_here * result[5];
    result[10] = result[5] + shift_to_here * result[4];
    result[9] = result[4] + shift_to_here * result[3];
    result[8] = result[3] + shift_to_here * result[2];
    result[7] = result[2] + shift_to_here * result[1];
    result[6] = result[1] + shift_to_here * result[0];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[17] = result[12] + shift_to_here * result[11];
    result[16] = result[11] + shift_to_here * result[10];
    result[15] = result[10] + shift_to_here * result[9];
    result[14] = result[9] + shift_to_here * result[8];
    result[13] = result[8] + shift_to_here * result[7];
    result[12] = result[7] + shift_to_here * result[6];
    result[27] = result[22] + shift_to_here * result[21];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[23] = result[18] + shift_to_here * result[17];
    result[22] = result[17] + shift_to_here * result[16];
    result[21] = result[16] + shift_to_here * result[15];
    result[20] = result[15] + shift_to_here * result[14];
    result[19] = result[14] + shift_to_here * result[13];
    result[18] = result[13] + shift_to_here * result[12];
    result[32] = result[27] + shift_to_here * result[26];
    result[31] = result[26] + shift_to_here * result[25];
    result[30] = result[25] + shift_to_here * result[24];
    result[29] = result[24] + shift_to_here * result[23];
    result[28] = result[23] + shift_to_here * result[22];
    result[27] = result[22] + shift_to_here * result[21];
    result[26] = result[21] + shift_to_here * result[20];
    result[25] = result[20] + shift_to_here * result[19];
    result[24] = result[19] + shift_to_here * result[18];
    result[37] = result[32] + shift_to_here * result[31];
    result[36] = result[31] + shift_to_here * result[30];
    result[35] = result[30] + shift_to_here * result[29];
    result[34] = result[29] + shift_to_here * result[28];
    result[33] = result[28] + shift_to_here * result[27];
    result[32] = result[27] + shift_to_here * result[26];
    result[31] = result[26] + shift_to_here * result[25];
    result[30] = result[25] + shift_to_here * result[24];
    result[42] = result[37] + shift_to_here * result[36];
    result[41] = result[36] + shift_to_here * result[35];
    result[40] = result[35] + shift_to_here * result[34];
    result[39] = result[34] + shift_to_here * result[33];
    result[38] = result[33] + shift_to_here * result[32];
    result[37] = result[32] + shift_to_here * result[31];
    result[36] = result[31] + shift_to_here * result[30];
    result[47] = result[42] + shift_to_here * result[41];
    result[46] = result[41] + shift_to_here * result[40];
    result[45] = result[40] + shift_to_here * result[39];
    result[44] = result[39] + shift_to_here * result[38];
    result[43] = result[38] + shift_to_here * result[37];
    result[42] = result[37] + shift_to_here * result[36];
  }
  if constexpr (i_angular == 7 && j_angular == 6) {
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[13] = result[7] + shift_to_here * result[6];
    result[12] = result[6] + shift_to_here * result[5];
    result[11] = result[5] + shift_to_here * result[4];
    result[10] = result[4] + shift_to_here * result[3];
    result[9] = result[3] + shift_to_here * result[2];
    result[8] = result[2] + shift_to_here * result[1];
    result[7] = result[1] + shift_to_here * result[0];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[20] = result[14] + shift_to_here * result[13];
    result[19] = result[13] + shift_to_here * result[12];
    result[18] = result[12] + shift_to_here * result[11];
    result[17] = result[11] + shift_to_here * result[10];
    result[16] = result[10] + shift_to_here * result[9];
    result[15] = result[9] + shift_to_here * result[8];
    result[14] = result[8] + shift_to_here * result[7];
    result[31] = result[25] + shift_to_here * result[24];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[27] = result[21] + shift_to_here * result[20];
    result[26] = result[20] + shift_to_here * result[19];
    result[25] = result[19] + shift_to_here * result[18];
    result[24] = result[18] + shift_to_here * result[17];
    result[23] = result[17] + shift_to_here * result[16];
    result[22] = result[16] + shift_to_here * result[15];
    result[21] = result[15] + shift_to_here * result[14];
    result[37] = result[31] + shift_to_here * result[30];
    result[36] = result[30] + shift_to_here * result[29];
    result[35] = result[29] + shift_to_here * result[28];
    result[34] = result[28] + shift_to_here * result[27];
    result[33] = result[27] + shift_to_here * result[26];
    result[32] = result[26] + shift_to_here * result[25];
    result[31] = result[25] + shift_to_here * result[24];
    result[30] = result[24] + shift_to_here * result[23];
    result[29] = result[23] + shift_to_here * result[22];
    result[28] = result[22] + shift_to_here * result[21];
    result[43] = result[37] + shift_to_here * result[36];
    result[42] = result[36] + shift_to_here * result[35];
    result[41] = result[35] + shift_to_here * result[34];
    result[40] = result[34] + shift_to_here * result[33];
    result[39] = result[33] + shift_to_here * result[32];
    result[38] = result[32] + shift_to_here * result[31];
    result[37] = result[31] + shift_to_here * result[30];
    result[36] = result[30] + shift_to_here * result[29];
    result[35] = result[29] + shift_to_here * result[28];
    result[49] = result[43] + shift_to_here * result[42];
    result[48] = result[42] + shift_to_here * result[41];
    result[47] = result[41] + shift_to_here * result[40];
    result[46] = result[40] + shift_to_here * result[39];
    result[45] = result[39] + shift_to_here * result[38];
    result[44] = result[38] + shift_to_here * result[37];
    result[43] = result[37] + shift_to_here * result[36];
    result[42] = result[36] + shift_to_here * result[35];
    result[55] = result[49] + shift_to_here * result[48];
    result[54] = result[48] + shift_to_here * result[47];
    result[53] = result[47] + shift_to_here * result[46];
    result[52] = result[46] + shift_to_here * result[45];
    result[51] = result[45] + shift_to_here * result[44];
    result[50] = result[44] + shift_to_here * result[43];
    result[49] = result[43] + shift_to_here * result[42];
  }
  if constexpr (i_angular == 7 && j_angular == 7) {
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[15] = result[8] + shift_to_here * result[7];
    result[14] = result[7] + shift_to_here * result[6];
    result[13] = result[6] + shift_to_here * result[5];
    result[12] = result[5] + shift_to_here * result[4];
    result[11] = result[4] + shift_to_here * result[3];
    result[10] = result[3] + shift_to_here * result[2];
    result[9] = result[2] + shift_to_here * result[1];
    result[8] = result[1] + shift_to_here * result[0];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[23] = result[16] + shift_to_here * result[15];
    result[22] = result[15] + shift_to_here * result[14];
    result[21] = result[14] + shift_to_here * result[13];
    result[20] = result[13] + shift_to_here * result[12];
    result[19] = result[12] + shift_to_here * result[11];
    result[18] = result[11] + shift_to_here * result[10];
    result[17] = result[10] + shift_to_here * result[9];
    result[16] = result[9] + shift_to_here * result[8];
    result[35] = result[28] + shift_to_here * result[27];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[31] = result[24] + shift_to_here * result[23];
    result[30] = result[23] + shift_to_here * result[22];
    result[29] = result[22] + shift_to_here * result[21];
    result[28] = result[21] + shift_to_here * result[20];
    result[27] = result[20] + shift_to_here * result[19];
    result[26] = result[19] + shift_to_here * result[18];
    result[25] = result[18] + shift_to_here * result[17];
    result[24] = result[17] + shift_to_here * result[16];
    result[42] = result[35] + shift_to_here * result[34];
    result[41] = result[34] + shift_to_here * result[33];
    result[40] = result[33] + shift_to_here * result[32];
    result[39] = result[32] + shift_to_here * result[31];
    result[38] = result[31] + shift_to_here * result[30];
    result[37] = result[30] + shift_to_here * result[29];
    result[36] = result[29] + shift_to_here * result[28];
    result[35] = result[28] + shift_to_here * result[27];
    result[34] = result[27] + shift_to_here * result[26];
    result[33] = result[26] + shift_to_here * result[25];
    result[32] = result[25] + shift_to_here * result[24];
    result[49] = result[42] + shift_to_here * result[41];
    result[48] = result[41] + shift_to_here * result[40];
    result[47] = result[40] + shift_to_here * result[39];
    result[46] = result[39] + shift_to_here * result[38];
    result[45] = result[38] + shift_to_here * result[37];
    result[44] = result[37] + shift_to_here * result[36];
    result[43] = result[36] + shift_to_here * result[35];
    result[42] = result[35] + shift_to_here * result[34];
    result[41] = result[34] + shift_to_here * result[33];
    result[40] = result[33] + shift_to_here * result[32];
    result[56] = result[49] + shift_to_here * result[48];
    result[55] = result[48] + shift_to_here * result[47];
    result[54] = result[47] + shift_to_here * result[46];
    result[53] = result[46] + shift_to_here * result[45];
    result[52] = result[45] + shift_to_here * result[44];
    result[51] = result[44] + shift_to_here * result[43];
    result[50] = result[43] + shift_to_here * result[42];
    result[49] = result[42] + shift_to_here * result[41];
    result[48] = result[41] + shift_to_here * result[40];
    result[63] = result[56] + shift_to_here * result[55];
    result[62] = result[55] + shift_to_here * result[54];
    result[61] = result[54] + shift_to_here * result[53];
    result[60] = result[53] + shift_to_here * result[52];
    result[59] = result[52] + shift_to_here * result[51];
    result[58] = result[51] + shift_to_here * result[50];
    result[57] = result[50] + shift_to_here * result[49];
    result[56] = result[49] + shift_to_here * result[48];
  }
}

template <int i_angular, int j_angular>
__forceinline__ __device__ void fill_with_recursion(
    double result[], const double prefactor, const double vrr_factor_prev,
    const double vrr_factor_second_prev, const double vrr_shift) {
  vertical_recursion<i_angular + j_angular>(result, prefactor, vrr_factor_prev,
                                            vrr_factor_second_prev);
  horizontal_recursion<i_angular, j_angular>(result, vrr_shift);
}
} // namespace rr
