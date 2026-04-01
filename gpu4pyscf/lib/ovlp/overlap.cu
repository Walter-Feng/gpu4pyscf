#include "macro.cuh"
#include "recursion.cuh"
#include "write.cuh"
#include <math.h>

namespace ovlp {
template <int i_angular, int j_angular>

__global__ void kernel(double *result, const int *pair_indices,
                       const int n_primitives, const int n_pairs,
                       const int *primitive_to_function, const int n_functions,
                       const int *atm, const int atm_stride, const int *bas,
                       const int bas_stride, const double *env,
                       const int env_stride, const int is_screened) {

  OVLP_SPELL;

  result += blockIdx.y * n_functions * n_functions +
            i_function_index * n_functions + j_function_index;

  if constexpr (i_angular == 0 && j_angular == 0) {
    atomicAdd(result, prefactor * prefactor * prefactor);
  } else {
    double x_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(x, 0, 0);

    double y_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(y, 0, 0);

    double z_pairs[(i_angular + 1) * (j_angular + 1)];
    reset(z, 0, 0);

    write(0);
  }
}

template <int i_angular, int j_angular>
__global__ void
gradient(double *result, const int *pair_indices, const int n_primitives,
         const int n_pairs, const int *primitive_to_function,
         const int n_functions, const int *atm, const int atm_stride,
         const int *bas, const int bas_stride, const double *env,
         const int env_stride, const int is_screened) {

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
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(x_pairs,
                                                                    2 * beta);
  write(1);
  reset(x, 0, 1);

  // // y component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(y_pairs,
                                                                    2 * beta);
  write(1);
  reset(y, 0, 1);

  // // z component
  rr::insert_gradient_operator<i_angular, j_angular, j_angular + 2>(z_pairs,
                                                                    2 * beta);
  write(1);
}
} // namespace ovlp

extern "C" {
void overlap(double *result, const int *pair_indices, const int n_pairs,
             const int n_primitives, const int *primitive_to_function,
             const int n_functions, const int *atm, const int atm_stride,
             const int *bas, const int bas_stride, const double *env,
             const int env_stride, const int n_configurations,
             const int i_angular, const int j_angular, const int is_screened) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) { tabulate_kernel(ovlp::kernel); }
}

void overlap_gradient(double *result, const int *pair_indices,
                      const int n_pairs, const int n_primitives,
                      const int *primitive_to_function, const int n_functions,
                      const int *atm, const int atm_stride, const int *bas,
                      const int bas_stride, const double *env,
                      const int env_stride, const int n_configurations,
                      const int i_angular, const int j_angular,
                      const int is_screened) {

  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) { tabulate_kernel(ovlp::gradient); }
}
}
