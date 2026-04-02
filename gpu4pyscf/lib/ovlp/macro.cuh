#pragma once

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

#define OVLP_SPELL                                                             \
  atm += blockIdx.y * atm_stride;                                              \
  bas += blockIdx.y * bas_stride;                                              \
  env += blockIdx.y * env_stride;                                              \
  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;                        \
  if (pair_idx >= n_pairs)                                                     \
    return;                                                                    \
  int i_primitive, j_primitive;                                                \
  if (is_screened) {                                                           \
    const int primitive_pair = pair_indices[pair_idx];                         \
    i_primitive = primitive_pair / n_primitives;                               \
    j_primitive = primitive_pair % n_primitives;                               \
  } else {                                                                     \
    const int bra_begin = pair_indices[0];                                     \
    const int bra_end = pair_indices[1];                                       \
    const int ket_begin = pair_indices[2];                                     \
    const int ket_end = pair_indices[3];                                       \
    const int n_rows = bra_end - bra_begin;                                    \
    const int n_cols = ket_end - ket_begin;                                    \
    if constexpr (i_angular == j_angular) {                                    \
      const double sqrt_target =                                               \
          (2 * n_cols + 1) * (2 * n_cols + 1) - 8 * pair_idx;                  \
      i_primitive = (int)floor((2 * n_cols - 1 - sqrt(sqrt_target)) / 2) + 1;  \
      j_primitive =                                                            \
          pair_idx - (2 * n_cols - i_primitive - 1) * i_primitive / 2;         \
    } else {                                                                   \
      const int stride = max(n_rows, n_cols);                                  \
      const int index_with_larger_stride = pair_idx / stride;                  \
      const int index_with_smaller_stride = pair_idx % stride;                 \
      i_primitive = n_rows >= n_cols ? index_with_smaller_stride               \
                                     : index_with_larger_stride;               \
      j_primitive = n_rows >= n_cols ? index_with_larger_stride                \
                                     : index_with_smaller_stride;              \
    }                                                                          \
    i_primitive += bra_begin;                                                  \
    j_primitive += ket_begin;                                                  \
  }                                                                            \
  const double alpha = env[bas(PTR_EXP, i_primitive)];                         \
  const double beta = env[bas(PTR_EXP, j_primitive)];                          \
  const double c1 =                                                            \
      env[bas(PTR_COEFF, i_primitive)] * rr::common_fac_sp<i_angular>();       \
  const double c2 =                                                            \
      env[bas(PTR_COEFF, j_primitive)] * rr::common_fac_sp<j_angular>();       \
  const int i_atom = bas(ATOM_OF, i_primitive);                                \
  const int j_atom = bas(ATOM_OF, j_primitive);                                \
  const int i_coord_offset = atm(PTR_COORD, i_atom);                           \
  const int j_coord_offset = atm(PTR_COORD, j_atom);                           \
  const double i_x = env[i_coord_offset + 0];                                  \
  const double i_y = env[i_coord_offset + 1];                                  \
  const double i_z = env[i_coord_offset + 2];                                  \
  const double j_x = env[j_coord_offset + 0];                                  \
  const double j_y = env[j_coord_offset + 1];                                  \
  const double j_z = env[j_coord_offset + 2];                                  \
  const double ix_to_jx = j_x - i_x;                                           \
  const double iy_to_jy = j_y - i_y;                                           \
  const double iz_to_jz = j_z - i_z;                                           \
  const double pair_distance_squared =                                         \
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;         \
  const double pair_exponent = alpha + beta;                                   \
                                                                               \
  double prefactor = sqrt(M_PI / pair_exponent);                               \
  prefactor *= prefactor * prefactor * c1 * c2;                                \
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);     \
  if (i_primitive == j_primitive) {                                            \
    prefactor *= 0.5;                                                          \
  }                                                                            \
  prefactor = cbrt(prefactor);                                                 \
                                                                               \
  const int i_function_index = primitive_to_function[i_primitive];             \
  const int j_function_index = primitive_to_function[j_primitive];             \
                                                                               \
  const double factor_a = -alpha / pair_exponent;                              \
  const double factor_b = 0.5 / pair_exponent;

#define reset(axis, bra_padding, ket_padding)                                  \
  rr::fill_with_recursion<i_angular + bra_padding, j_angular + ket_padding>(   \
      axis##_pairs, prefactor, factor_a * i##axis##_to_j##axis, factor_b,      \
      i##axis##_to_j##axis);

#define write(stride_padding)                                                  \
  write_integral<i_angular, j_angular, j_angular + 1 + stride_padding>(        \
      result, x_pairs, y_pairs, z_pairs, n_functions);                         \
  result += n_functions * n_functions;

#define r(axis, bra_padding, ket_padding)                                      \
  rr::insert_position_operator<i_angular + bra_padding,                        \
                               j_angular + ket_padding,                        \
                               j_angular + 2 + ket_padding>(                   \
      axis##_pairs, j_##axis - reference_point_##axis);

#define p(axis, bra_padding, ket_padding)                                      \
  rr::insert_gradient_operator_to_bra<i_angular + bra_padding,                 \
                                      j_angular + ket_padding,                 \
                                      j_angular + 1 + ket_padding>(            \
      axis##_pairs, 2 * alpha);

// kernel macro
#define kernel_macro(kernel, i, j)                                             \
  case i * 10 + j:                                                             \
    kernel<i, j><<<block_grid, block_size>>>(                                  \
        result, pair_indices, n_primitives, n_pairs, primitive_to_function,    \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride,        \
        is_screened);                                                          \
    break;

#define multipole_kernel_macro(kernel, i, j)                                   \
  case i * 10 + j:                                                             \
    kernel<i, j><<<block_grid, block_size>>>(                                  \
        result, pair_indices, n_primitives, n_pairs, primitive_to_function,    \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride,        \
        reference_point_x, reference_point_y, reference_point_z, is_screened); \
    break;

// tabulator
#define tabulate_multipole(kernel)                                             \
  multipole_kernel_macro(kernel, 0, 0);                                        \
  multipole_kernel_macro(kernel, 0, 1);                                        \
  multipole_kernel_macro(kernel, 0, 2);                                        \
  multipole_kernel_macro(kernel, 0, 3);                                        \
  multipole_kernel_macro(kernel, 0, 4);                                        \
  multipole_kernel_macro(kernel, 1, 1);                                        \
  multipole_kernel_macro(kernel, 1, 2);                                        \
  multipole_kernel_macro(kernel, 1, 3);                                        \
  multipole_kernel_macro(kernel, 1, 4);                                        \
  multipole_kernel_macro(kernel, 2, 2);                                        \
  multipole_kernel_macro(kernel, 2, 3);                                        \
  multipole_kernel_macro(kernel, 2, 4);                                        \
  multipole_kernel_macro(kernel, 3, 3);                                        \
  multipole_kernel_macro(kernel, 3, 4);                                        \
  multipole_kernel_macro(kernel, 4, 4);

#define tabulate_kernel(kernel)                                                \
  kernel_macro(kernel, 0, 0);                                                  \
  kernel_macro(kernel, 0, 1);                                                  \
  kernel_macro(kernel, 0, 2);                                                  \
  kernel_macro(kernel, 0, 3);                                                  \
  kernel_macro(kernel, 0, 4);                                                  \
  kernel_macro(kernel, 1, 1);                                                  \
  kernel_macro(kernel, 1, 2);                                                  \
  kernel_macro(kernel, 1, 3);                                                  \
  kernel_macro(kernel, 1, 4);                                                  \
  kernel_macro(kernel, 2, 2);                                                  \
  kernel_macro(kernel, 2, 3);                                                  \
  kernel_macro(kernel, 2, 4);                                                  \
  kernel_macro(kernel, 3, 3);                                                  \
  kernel_macro(kernel, 3, 4);                                                  \
  kernel_macro(kernel, 4, 4);
