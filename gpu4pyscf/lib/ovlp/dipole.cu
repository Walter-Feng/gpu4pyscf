#include "macro.cuh"
#include "recursion.cuh"
#include "write.cuh"
#include <math.h>

namespace ovlp {
template <int i_angular, int j_angular>
__global__ void
dipole_kernel(double *result, const int *pair_indices, const int n_primitives,
              const int n_pairs, const int *primitive_to_function,
              const int n_functions, const int *atm, const int atm_stride,
              const int *bas, const int bas_stride, const double *env,
              const int env_stride, const double reference_point_x,
              const double reference_point_y, const double reference_point_z,
              const int is_screened) {

  OVLP_SPELL;

  result += blockIdx.y * 3 * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(x, 0, 1);

  double y_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(y, 0, 1);

  double z_pairs[(i_angular + 1) * (j_angular + 2)];
  reset(z, 0, 1);

  // // x component
  rr::insert_position_operator<i_angular, j_angular, j_angular + 2>(
      x_pairs, j_x - reference_point_x);

  write(1);
  reset(x, 0, 1);

  // // y component
  rr::insert_position_operator<i_angular, j_angular, j_angular + 2>(
      y_pairs, j_y - reference_point_y);
  write(1);
  reset(y, 0, 1);

  // // z component
  rr::insert_position_operator<i_angular, j_angular, j_angular + 2>(
      z_pairs, j_z - reference_point_z);
  write(1);
}

template <int i_angular, int j_angular>
__global__ void
dipole_gradient(double *dipole_gradient, const int *pair_indices,
                const int n_primitives, const int n_pairs,
                const int *primitive_to_function, const int n_functions,
                const int *atm, const int atm_stride, const int *bas,
                const int bas_stride, const double *env, const int env_stride,
                const double reference_point_x, const double reference_point_y,
                const double reference_point_z, const int is_screened) {

  OVLP_SPELL;

  dipole_gradient += blockIdx.y * 9 * n_functions * n_functions +
                     i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 2) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular + 1, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);

  double y_pairs[(i_angular + 2) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular + 1, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  double z_pairs[(i_angular + 2) * (j_angular + 2)];
  rr::fill_with_recursion<i_angular + 1, j_angular + 1>(
      z_pairs, 1, factor_a * iz_to_jz, factor_b, iz_to_jz);

  // partial x component

  // // x component
  rr::insert_position_operator<i_angular + 1, j_angular, j_angular + 2>(
      x_pairs, j_x - reference_point_x);
  rr::insert_gradient_operator_to_bra<i_angular, j_angular, j_angular + 2>(
      x_pairs, 2 * alpha);
  write_integral<i_angular, j_angular, j_angular + 2>(
      dipole_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular + 1, j_angular + 1>(
      x_pairs, prefactor, factor_a * ix_to_jx, factor_b, ix_to_jx);
  rr::insert_gradient_operator_to_bra<i_angular, j_angular, j_angular + 2>(
      x_pairs, 2 * alpha);

  dipole_gradient += n_functions * n_functions;

  // // y component
  rr::insert_position_operator<i_angular, j_angular, j_angular + 2>(
      y_pairs, j_y - reference_point_y);
  write_integral<i_angular, j_angular, j_angular + 2>(
      dipole_gradient, x_pairs, y_pairs, z_pairs, n_functions);
  rr::fill_with_recursion<i_angular + 1, j_angular + 1>(
      y_pairs, 1, factor_a * iy_to_jy, factor_b, iy_to_jy);

  dipole_gradient += n_functions * n_functions;

  // // z component
  rr::insert_position_operator<i_angular, j_angular, j_angular + 2>(
      z_pairs, j_z - reference_point_z);
  write_integral<i_angular, j_angular, j_angular + 2>(
      dipole_gradient, x_pairs, y_pairs, z_pairs, n_functions);

  dipole_gradient += n_functions * n_functions;

  rr::insert_position_operator<i_angular + 1, j_angular, j_angular + 2>(
      x_pairs, j_x - reference_point_x);

  // partial y component
}
} // namespace ovlp

extern "C" {
void dipole(double *result, const int *pair_indices, const int n_pairs,
            const int n_primitives, const int *primitive_to_function,
            const int n_functions, const int *atm, const int atm_stride,
            const int *bas, const int bas_stride, const double *env,
            const int env_stride, const int n_configurations,
            const int i_angular, const int j_angular,
            const double reference_point_x, const double reference_point_y,
            const double reference_point_z, const int is_screened) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    tabulate_multipole(ovlp::dipole_kernel);
  }
}
}
