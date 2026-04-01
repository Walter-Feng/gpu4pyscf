#include "macro.cuh"
#include "recursion.cuh"
#include "write.cuh"
#include <math.h>

namespace ovlp {
template <int i_angular, int j_angular>
__global__ void quadrupole_kernel(
    double *result, const int *pair_indices, const int n_primitives,
    const int n_pairs, const int *primitive_to_function, const int n_functions,
    const int *atm, const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const double reference_point_x,
    const double reference_point_y, const double reference_point_z,
    const int is_screened) {

  OVLP_SPELL;

  result += blockIdx.y * 9 * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 1) * (j_angular + 3)];
  reset(x, 0, 2);

  double y_pairs[(i_angular + 1) * (j_angular + 3)];
  reset(y, 0, 2);

  double z_pairs[(i_angular + 1) * (j_angular + 3)];
  reset(z, 0, 2);

  // x^2 component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      x_pairs, j_x - reference_point_x);
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      x_pairs, j_x - reference_point_x);
  write(2);
  reset(x, 0, 2);

  // xy component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      x_pairs, j_x - reference_point_x);
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      y_pairs, j_y - reference_point_y);
  write(2);
  reset(y, 0, 2);

  // xz component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      z_pairs, j_z - reference_point_z);
  write(2);
  reset(x, 0, 2);

  result += 2 * n_functions * n_functions;

  // yz component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      y_pairs, j_y - reference_point_y);
  write(2);
  reset(y, 0, 2);

  result += 2 * n_functions * n_functions;

  // z^2 component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      z_pairs, j_z - reference_point_z);
  write(2);
  reset(z, 0, 2);
  result -= 5 * n_functions * n_functions;

  // y^2 component
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      y_pairs, j_y - reference_point_y);
  rr::insert_position_operator<i_angular, j_angular + 1, j_angular + 3>(
      y_pairs, j_y - reference_point_y);
  write(2);
}
} // namespace ovlp
extern "C" {
void quadrupole(double *result, const int *pair_indices, const int n_pairs,
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
    tabulate_multipole(ovlp::quadrupole_kernel);
  }
}
}
