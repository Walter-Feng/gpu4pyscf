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
  r(x, 0, 1);
  r(x, 0, 1);
  write(2);
  reset(x, 0, 2);

  // xy component
  r(x, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(y, 0, 2);

  // xz component
  r(z, 0, 1);
  write(2);
  reset(x, 0, 2);

  result += 2 * n_functions * n_functions;

  // yz component
  r(y, 0, 1);
  write(2);
  reset(y, 0, 2);

  result += 2 * n_functions * n_functions;

  // z^2 component
  r(z, 0, 1);
  write(2);
  reset(z, 0, 2);
  result -= 5 * n_functions * n_functions;

  // y^2 component
  r(y, 0, 1);
  r(y, 0, 1);
  write(2);
}

template <int i_angular, int j_angular>
__global__ void quadrupole_gradient(
    double *result, const int *pair_indices, const int n_primitives,
    const int n_pairs, const int *primitive_to_function, const int n_functions,
    const int *atm, const int atm_stride, const int *bas, const int bas_stride,
    const double *env, const int env_stride, const double reference_point_x,
    const double reference_point_y, const double reference_point_z,
    const int is_screened) {

  OVLP_SPELL;

  result += blockIdx.y * 27 * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  double x_pairs[(i_angular + 2) * (j_angular + 3)];
  reset(x, 1, 2);

  double y_pairs[(i_angular + 2) * (j_angular + 3)];
  reset(y, 1, 2);

  double z_pairs[(i_angular + 2) * (j_angular + 3)];
  reset(z, 1, 2);

  // partial x component
  // x^2 component
  p(x, 0, 2);
  r(x, 0, 1);
  r(x, 0, 1);
  write(2);
  reset(x, 1, 2);
  p(x, 0, 2);

  // xy component
  r(x, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);

  // xz component
  r(z, 0, 1);
  write(2);
  reset(x, 1, 2);
  p(x, 0, 2);

  result += 2 * n_functions * n_functions;

  // yz component
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);

  result += 2 * n_functions * n_functions;

  // z^2 component
  r(z, 0, 1);
  write(2);
  reset(z, 1, 2);
  result -= 5 * n_functions * n_functions;

  // y^2 component
  r(y, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(x, 1, 2);
  reset(y, 1, 2);

  result += 4 * n_functions * n_functions;

  // partial y component

  p(y, 0, 2);

  // x^2 component
  r(x, 0, 1);
  r(x, 0, 1);
  write(2);
  reset(x, 1, 2);

  // xy component
  r(x, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);
  p(y, 0, 2);

  // xz component
  r(z, 0, 1);
  write(2);
  reset(x, 1, 2);

  result += 2 * n_functions * n_functions;

  // yz component
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);
  p(y, 0, 2);

  result += 2 * n_functions * n_functions;

  // z^2 component
  r(z, 0, 1);
  write(2);
  reset(z, 1, 2);
  result -= 5 * n_functions * n_functions;

  // y^2 component
  r(y, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);

  result += 4 * n_functions * n_functions;

  // partial z component

  p(z, 0, 2);

  // x^2 component
  r(x, 0, 1);
  r(x, 0, 1);
  write(2);
  reset(x, 1, 2);

  // xy component
  r(x, 0, 1);
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);

  // xz component
  r(z, 0, 1);
  write(2);
  reset(x, 1, 2);

  result += 2 * n_functions * n_functions;

  // yz component
  r(y, 0, 1);
  write(2);
  reset(y, 1, 2);

  result += 2 * n_functions * n_functions;

  // z^2 component
  r(z, 0, 1);
  write(2);
  reset(z, 1, 2);
  p(z, 0, 2);
  result -= 5 * n_functions * n_functions;

  // y^2 component
  r(y, 0, 1);
  r(y, 0, 1);
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

void quadrupole_gradient(
    double *result, const int *pair_indices, const int n_pairs,
    const int n_primitives, const int *primitive_to_function,
    const int n_functions, const int *atm, const int atm_stride, const int *bas,
    const int bas_stride, const double *env, const int env_stride,
    const int n_configurations, const int i_angular, const int j_angular,
    const double reference_point_x, const double reference_point_y,
    const double reference_point_z, const int is_screened) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    tabulate_multipole(ovlp::quadrupole_gradient);
  }
}
}
