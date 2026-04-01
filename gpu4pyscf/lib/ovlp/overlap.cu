// slots of atm
#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS 6

// slots of bas
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define KAPPA_OF 4
#define PTR_EXP 5
#define PTR_COEFF 6
#define PTR_BAS_COORD 7
#define BAS_SLOTS 8

#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]

#include "recursion.cuh"
#include "write.cuh"
#include <math.h>
#include <stdio.h>

namespace ovlp {
template <int i_angular, int j_angular>
__global__ void kernel(double *ovlp, const int *pair_indices,
                       const int n_primitives, const int n_pairs,
                       const int *primitive_to_function, const int n_functions,
                       const int *atm, const int atm_stride, const int *bas,
                       const int bas_stride, const double *env,
                       const int env_stride) {
  atm += blockIdx.y * atm_stride;
  bas += blockIdx.y * bas_stride;
  env += blockIdx.y * env_stride;

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= n_pairs)
    return;

  const int primitive_pair = pair_indices[pair_idx];
  const int i_primitive = primitive_pair / n_primitives;
  const int j_primitive = primitive_pair % n_primitives;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];

  const double c1 =
      env[bas(PTR_COEFF, i_primitive)] * rr::common_fac_sp<i_angular>();
  const double c2 =
      env[bas(PTR_COEFF, j_primitive)] * rr::common_fac_sp<j_angular>();

  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);

  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);

  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];

  const double j_x = env[j_coord_offset + 0];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];

  const double ix_to_jx = j_x - i_x;
  const double iy_to_jy = j_y - i_y;
  const double iz_to_jz = j_z - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;

  const double pair_exponent = alpha + beta;
  double prefactor = sqrt(M_PI / pair_exponent);
  prefactor *= prefactor * prefactor * c1 * c2;
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);
  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];

  ovlp += blockIdx.y * n_functions * n_functions +
          i_function_index * n_functions + j_function_index;

  if constexpr (i_angular == 0 && j_angular == 0) {
    atomicAdd(ovlp, prefactor);
  } else {
    const double factor_a = -alpha / pair_exponent;
    const double factor_b = 0.5 / pair_exponent;

    double x_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        x_pairs, prefactor, factor_a * ix_to_jx, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(x_pairs, ix_to_jx);

    double y_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        y_pairs, 1, factor_a * iy_to_jy, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(y_pairs, iy_to_jy);

    double z_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        z_pairs, 1, factor_a * iz_to_jz, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(z_pairs, iz_to_jz);

    write_integral<i_angular, j_angular, j_angular + 1>(ovlp, x_pairs, y_pairs,
                                                        z_pairs, n_functions);
  }
}

template <int i_angular, int j_angular>
__global__ void kernel(double *ovlp, const int bra_begin, const int bra_end,
                       const int ket_begin, const int ket_end,
                       const int *primitive_to_function, const int n_functions,
                       const int *atm, const int atm_stride, const int *bas,
                       const int bas_stride, const double *env,
                       const int env_stride) {
  atm += blockIdx.y * atm_stride;
  bas += blockIdx.y * bas_stride;
  env += blockIdx.y * env_stride;

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = bra_end - bra_begin;
  const int n_cols = ket_end - ket_begin;
  int n_pairs, i_primitive, j_primitive;

  if constexpr (i_angular == j_angular) {
    n_pairs = n_rows * (n_rows + 1) / 2;
    const float sqrt_target =
        (2 * n_cols + 1) * (2 * n_cols + 1) - 8 * pair_idx;
    i_primitive = (int)floor((2 * n_cols - 1 - sqrt(sqrt_target)) / 2) + 1;
    j_primitive = pair_idx - (2 * n_cols - i_primitive - 1) * i_primitive / 2;
  } else {
    n_pairs = n_rows * n_cols;
    const int stride = max(n_rows, n_cols);
    const int index_with_larger_stride = pair_idx / stride;
    const int index_with_smaller_stride = pair_idx % stride;
    i_primitive =
        n_rows >= n_cols ? index_with_smaller_stride : index_with_larger_stride;
    j_primitive =
        n_rows >= n_cols ? index_with_larger_stride : index_with_smaller_stride;
  }
  i_primitive += bra_begin;
  j_primitive += ket_begin;

  if (pair_idx >= n_pairs)
    return;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];

  const double c1 =
      env[bas(PTR_COEFF, i_primitive)] * rr::common_fac_sp<i_angular>();
  const double c2 =
      env[bas(PTR_COEFF, j_primitive)] * rr::common_fac_sp<j_angular>();

  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);

  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);

  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];

  const double j_x = env[j_coord_offset + 0];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];

  const double ix_to_jx = j_x - i_x;
  const double iy_to_jy = j_y - i_y;
  const double iz_to_jz = j_z - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;

  const double pair_exponent = alpha + beta;
  double prefactor = sqrt(M_PI / pair_exponent);
  prefactor *= prefactor * prefactor * c1 * c2;
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);
  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];

  ovlp += blockIdx.y * n_functions * n_functions +
          i_function_index * n_functions + j_function_index;

  if constexpr (i_angular == 0 && j_angular == 0) {
    atomicAdd(ovlp, prefactor);
  } else {
    const double factor_a = -alpha / pair_exponent;
    const double factor_b = 0.5 / pair_exponent;

    double x_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        x_pairs, prefactor, factor_a * ix_to_jx, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(x_pairs, ix_to_jx);

    double y_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        y_pairs, 1, factor_a * iy_to_jy, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(y_pairs, iy_to_jy);

    double z_pairs[(i_angular + 1) * (j_angular + 1)];
    rr::vertical_recursion<i_angular + j_angular>(
        z_pairs, 1, factor_a * iz_to_jz, factor_b);
    rr::horizontal_recursion<i_angular, j_angular>(z_pairs, iz_to_jz);

    write_integral<i_angular, j_angular, j_angular + 1>(ovlp, x_pairs, y_pairs,
                                                        z_pairs, n_functions);
  }
}

template <int i_angular, int j_angular>
__global__ void
gradient(double *ovlp_gradient, const int *pair_indices, const int n_primitives,
         const int n_pairs, const int *primitive_to_function,
         const int n_functions, const int *atm, const int atm_stride,
         const int *bas, const int bas_stride, const double *env,
         const int env_stride) {
  atm += blockIdx.y * atm_stride;
  bas += blockIdx.y * bas_stride;
  env += blockIdx.y * env_stride;

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= n_pairs)
    return;

  const int primitive_pair = pair_indices[pair_idx];
  const int i_primitive = primitive_pair / n_primitives;
  const int j_primitive = primitive_pair % n_primitives;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];

  const double c1 =
      env[bas(PTR_COEFF, i_primitive)] * rr::common_fac_sp<i_angular>();
  const double c2 =
      env[bas(PTR_COEFF, j_primitive)] * rr::common_fac_sp<j_angular>();

  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);

  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);

  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];

  const double j_x = env[j_coord_offset + 0];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];

  const double ix_to_jx = j_x - i_x;
  const double iy_to_jy = j_y - i_y;
  const double iz_to_jz = j_z - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;

  const double pair_exponent = alpha + beta;
  double prefactor = sqrt(M_PI / pair_exponent);
  prefactor *= prefactor * prefactor * c1 * c2;
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);
  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];

  ovlp_gradient += blockIdx.y * n_functions * n_functions +
                   i_function_index * n_functions + j_function_index;

  const double factor_a = -alpha / pair_exponent;
  const double factor_b = 0.5 / pair_exponent;

  double x_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);

  double y_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  double z_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      z_pairs, 1, factor_a * iz_to_jz, factor_b, iz_to_jz);

  // // x component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(x_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);

  ovlp_gradient += n_functions * n_functions;

  // // y component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(y_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  ovlp_gradient += n_functions * n_functions;

  // // z component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(z_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
}

template <int i_angular, int j_angular>
__global__ void
gradient(double *ovlp_gradient, const int bra_begin, const int bra_end,
         const int ket_begin, const int ket_end,
         const int *primitive_to_function, const int n_functions,
         const int *atm, const int atm_stride, const int *bas,
         const int bas_stride, const double *env, const int env_stride) {
  atm += blockIdx.y * atm_stride;
  bas += blockIdx.y * bas_stride;
  env += blockIdx.y * env_stride;

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = bra_end - bra_begin;
  const int n_cols = ket_end - ket_begin;
  int n_pairs, i_primitive, j_primitive;

  if constexpr (i_angular == j_angular) {
    n_pairs = n_rows * (n_rows + 1) / 2;
    const float sqrt_target =
        (2 * n_cols + 1) * (2 * n_cols + 1) - 8 * pair_idx;
    i_primitive = floor((2 * n_cols - 1 - sqrt(sqrt_target)) / 2) + 1;
    j_primitive = pair_idx - (2 * n_cols - i_primitive - 1) * i_primitive / 2;
  } else {
    n_pairs = n_rows * n_cols;
    const int stride = max(n_rows, n_cols);
    const int index_with_larger_stride = pair_idx / stride;
    const int index_with_smaller_stride = pair_idx % stride;
    i_primitive =
        n_rows >= n_cols ? index_with_smaller_stride : index_with_larger_stride;
    j_primitive =
        n_rows >= n_cols ? index_with_larger_stride : index_with_smaller_stride;
  }
  i_primitive += bra_begin;
  j_primitive += ket_begin;

  if (pair_idx >= n_pairs)
    return;

  const double alpha = env[bas(PTR_EXP, i_primitive)];
  const double beta = env[bas(PTR_EXP, j_primitive)];

  const double c1 =
      env[bas(PTR_COEFF, i_primitive)] * rr::common_fac_sp<i_angular>();
  const double c2 =
      env[bas(PTR_COEFF, j_primitive)] * rr::common_fac_sp<j_angular>();

  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);

  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);

  const double i_x = env[i_coord_offset + 0];
  const double i_y = env[i_coord_offset + 1];
  const double i_z = env[i_coord_offset + 2];

  const double j_x = env[j_coord_offset + 0];
  const double j_y = env[j_coord_offset + 1];
  const double j_z = env[j_coord_offset + 2];

  const double ix_to_jx = j_x - i_x;
  const double iy_to_jy = j_y - i_y;
  const double iz_to_jz = j_z - i_z;
  const double pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;

  const double pair_exponent = alpha + beta;
  double prefactor = sqrt(M_PI / pair_exponent);
  prefactor *= prefactor * prefactor * c1 * c2;
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);
  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];

  ovlp_gradient += blockIdx.y * n_functions * n_functions +
                   i_function_index * n_functions + j_function_index;

  const double factor_a = -alpha / pair_exponent;
  const double factor_b = 0.5 / pair_exponent;

  double x_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);

  double y_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  double z_pairs[(i_angular + 1) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      z_pairs, 1, factor_a * iz_to_jz, factor_b, iz_to_jz);

  // // x component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(x_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);

  ovlp_gradient += n_functions * n_functions;

  // // y component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(y_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  ovlp_gradient += n_functions * n_functions;

  // // z component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(z_pairs,
                                                                    2 * beta);
  write_integral<i_angular, j_angular, j_angular + 2>(
      ovlp_gradient, x_pairs, y_pairs, z_pairs, n_functions);
}
} // namespace ovlp

#define ovlp_kernel_macro(i, j)                                                \
  case i * 10 + j:                                                             \
    ovlp::kernel<i, j><<<block_grid, block_size>>>(                            \
        ovlp, pair_indices, n_primitives, n_pairs, primitive_to_function,      \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride);       \
    break;

#define ovlp_without_screening_kernel_macro(i, j)                              \
  case i * 10 + j:                                                             \
    ovlp::kernel<i, j><<<block_grid, block_size>>>(                            \
        ovlp, bra_begin, bra_end, ket_begin, ket_end, primitive_to_function,   \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride);       \
    break;

#define ovlp_gradient_kernel_macro(i, j)                                       \
  case i * 10 + j:                                                             \
    ovlp::gradient<i, j><<<block_grid, block_size>>>(                          \
        ovlp_gradient, pair_indices, n_primitives, n_pairs,                    \
        primitive_to_function, n_functions, atm, atm_stride, bas, bas_stride,  \
        env, env_stride);                                                      \
    break;

#define ovlp_gradient_without_screening_kernel_macro(i, j)                     \
  case i * 10 + j:                                                             \
    ovlp::gradient<i, j><<<block_grid, block_size>>>(                          \
        ovlp, bra_begin, bra_end, ket_begin, ket_end, primitive_to_function,   \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride);       \
    break;

extern "C" {
void overlap(double *ovlp, const int *pair_indices, const int n_pairs,
             const int n_primitives, const int *primitive_to_function,
             const int n_functions, const int *atm, const int atm_stride,
             const int *bas, const int bas_stride, const double *env,
             const int env_stride, const int n_configurations,
             const int i_angular, const int j_angular) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    ovlp_kernel_macro(0, 0);
    ovlp_kernel_macro(0, 1);
    ovlp_kernel_macro(0, 2);
    ovlp_kernel_macro(0, 3);
    ovlp_kernel_macro(0, 4);

    ovlp_kernel_macro(1, 0);
    ovlp_kernel_macro(1, 1);
    ovlp_kernel_macro(1, 2);
    ovlp_kernel_macro(1, 3);
    ovlp_kernel_macro(1, 4);

    ovlp_kernel_macro(2, 0);
    ovlp_kernel_macro(2, 1);
    ovlp_kernel_macro(2, 2);
    ovlp_kernel_macro(2, 3);
    ovlp_kernel_macro(2, 4);

    ovlp_kernel_macro(3, 0);
    ovlp_kernel_macro(3, 1);
    ovlp_kernel_macro(3, 2);
    ovlp_kernel_macro(3, 3);
    ovlp_kernel_macro(3, 4);

    ovlp_kernel_macro(4, 0);
    ovlp_kernel_macro(4, 1);
    ovlp_kernel_macro(4, 2);
    ovlp_kernel_macro(4, 3);
    ovlp_kernel_macro(4, 4);
  }
}

void overlap_without_screening(
    double *ovlp, const int bra_begin, const int bra_end, const int ket_begin,
    const int ket_end, const int *primitive_to_function, const int n_functions,
    const int *atm, const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const int n_configurations,
    const int i_angular, const int j_angular) {

  const int n_rows = bra_end - bra_begin;
  const int n_cols = ket_end - ket_begin;
  int n_pairs;
  if (i_angular == j_angular) {
    n_pairs = n_rows * (n_rows + 1) / 2;
  } else {
    n_pairs = n_rows * n_cols;
  }
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{static_cast<uint>((n_pairs + 255) / 256),
                        (uint)n_configurations, 1};

  switch (i_angular * 10 + j_angular) {
    ovlp_without_screening_kernel_macro(0, 0);
    ovlp_without_screening_kernel_macro(0, 1);
    ovlp_without_screening_kernel_macro(0, 2);
    ovlp_without_screening_kernel_macro(0, 3);
    ovlp_without_screening_kernel_macro(0, 4);

    ovlp_without_screening_kernel_macro(1, 0);
    ovlp_without_screening_kernel_macro(1, 1);
    ovlp_without_screening_kernel_macro(1, 2);
    ovlp_without_screening_kernel_macro(1, 3);
    ovlp_without_screening_kernel_macro(1, 4);

    ovlp_without_screening_kernel_macro(2, 0);
    ovlp_without_screening_kernel_macro(2, 1);
    ovlp_without_screening_kernel_macro(2, 2);
    ovlp_without_screening_kernel_macro(2, 3);
    ovlp_without_screening_kernel_macro(2, 4);

    ovlp_without_screening_kernel_macro(3, 0);
    ovlp_without_screening_kernel_macro(3, 1);
    ovlp_without_screening_kernel_macro(3, 2);
    ovlp_without_screening_kernel_macro(3, 3);
    ovlp_without_screening_kernel_macro(3, 4);

    ovlp_without_screening_kernel_macro(4, 0);
    ovlp_without_screening_kernel_macro(4, 1);
    ovlp_without_screening_kernel_macro(4, 2);
    ovlp_without_screening_kernel_macro(4, 3);
    ovlp_without_screening_kernel_macro(4, 4);
  }
}

void overlap_gradient(double *ovlp_gradient, const int *pair_indices,
                      const int n_pairs, const int n_primitives,
                      const int *primitive_to_function, const int n_functions,
                      const int *atm, const int atm_stride, const int *bas,
                      const int bas_stride, const double *env,
                      const int env_stride, const int n_configurations,
                      const int i_angular, const int j_angular) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    ovlp_gradient_kernel_macro(0, 0);
    ovlp_gradient_kernel_macro(0, 1);
    ovlp_gradient_kernel_macro(0, 2);
    ovlp_gradient_kernel_macro(0, 3);
    ovlp_gradient_kernel_macro(0, 4);

    ovlp_gradient_kernel_macro(1, 0);
    ovlp_gradient_kernel_macro(1, 1);
    ovlp_gradient_kernel_macro(1, 2);
    ovlp_gradient_kernel_macro(1, 3);
    ovlp_gradient_kernel_macro(1, 4);

    ovlp_gradient_kernel_macro(2, 0);
    ovlp_gradient_kernel_macro(2, 1);
    ovlp_gradient_kernel_macro(2, 2);
    ovlp_gradient_kernel_macro(2, 3);
    ovlp_gradient_kernel_macro(2, 4);

    ovlp_gradient_kernel_macro(3, 0);
    ovlp_gradient_kernel_macro(3, 1);
    ovlp_gradient_kernel_macro(3, 2);
    ovlp_gradient_kernel_macro(3, 3);
    ovlp_gradient_kernel_macro(3, 4);

    ovlp_gradient_kernel_macro(4, 0);
    ovlp_gradient_kernel_macro(4, 1);
    ovlp_gradient_kernel_macro(4, 2);
    ovlp_gradient_kernel_macro(4, 3);
    ovlp_gradient_kernel_macro(4, 4);
  }
}

void overlap_gradient_without_screening(
    double *ovlp, const int bra_begin, const int bra_end, const int ket_begin,
    const int ket_end, const int *primitive_to_function, const int n_functions,
    const int *atm, const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const int n_configurations,
    const int i_angular, const int j_angular) {

  const int n_rows = bra_end - bra_begin;
  const int n_cols = ket_end - ket_begin;
  int n_pairs;
  if (i_angular == j_angular) {
    n_pairs = n_rows * (n_rows + 1) / 2;
  } else {
    n_pairs = n_rows * n_cols;
  }
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{static_cast<uint>((n_pairs + 255) / 256),
                        (uint)n_configurations, 1};

  switch (i_angular * 10 + j_angular) {
    ovlp_gradient_without_screening_kernel_macro(0, 0);
    ovlp_gradient_without_screening_kernel_macro(0, 1);
    ovlp_gradient_without_screening_kernel_macro(0, 2);
    ovlp_gradient_without_screening_kernel_macro(0, 3);
    ovlp_gradient_without_screening_kernel_macro(0, 4);

    ovlp_gradient_without_screening_kernel_macro(1, 0);
    ovlp_gradient_without_screening_kernel_macro(1, 1);
    ovlp_gradient_without_screening_kernel_macro(1, 2);
    ovlp_gradient_without_screening_kernel_macro(1, 3);
    ovlp_gradient_without_screening_kernel_macro(1, 4);

    ovlp_gradient_without_screening_kernel_macro(2, 0);
    ovlp_gradient_without_screening_kernel_macro(2, 1);
    ovlp_gradient_without_screening_kernel_macro(2, 2);
    ovlp_gradient_without_screening_kernel_macro(2, 3);
    ovlp_gradient_without_screening_kernel_macro(2, 4);

    ovlp_gradient_without_screening_kernel_macro(3, 0);
    ovlp_gradient_without_screening_kernel_macro(3, 1);
    ovlp_gradient_without_screening_kernel_macro(3, 2);
    ovlp_gradient_without_screening_kernel_macro(3, 3);
    ovlp_gradient_without_screening_kernel_macro(3, 4);

    ovlp_gradient_without_screening_kernel_macro(4, 0);
    ovlp_gradient_without_screening_kernel_macro(4, 1);
    ovlp_gradient_without_screening_kernel_macro(4, 2);
    ovlp_gradient_without_screening_kernel_macro(4, 3);
    ovlp_gradient_without_screening_kernel_macro(4, 4);
  }
}
}
