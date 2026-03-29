#pragma once

namespace ovlp {
template <int i_angular, int j_angular, int stride>
__forceinline__ __device__ void
write_integral(double *output, const double x_pairs[], const double y_pairs[],
               const double z_pairs[], const int n_functions) {

  double expression;
  if constexpr (i_angular == 0 && j_angular == 0) {
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
  }
  if constexpr (i_angular == 0 && j_angular == 1) {
    expression = x_pairs[0 * stride + 1] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 1] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 2, expression);
  }
  if constexpr (i_angular == 0 && j_angular == 2) {
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.31539156525252 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.5462742152960396 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 4, expression);
  }
  if constexpr (i_angular == 0 && j_angular == 3) {
    expression = 1.7701307697799304 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -1.1195289977703462 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 1.4453057213202771 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = 0.5900435899266435 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 6, expression);
  }
  if constexpr (i_angular == 0 && j_angular == 4) {
    expression = 2.5033429417967046 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.94617469575756 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.31735664074561293 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = -0.47308734787878 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 5.310392309339791 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 7, expression);
    expression = 0.6258357354491761 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 8, expression);
  }
  if constexpr (i_angular == 1 && j_angular == 0) {
    expression = x_pairs[1 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[1 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
  }
  if constexpr (i_angular == 1 && j_angular == 1) {
    expression = x_pairs[1 * stride + 1] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = x_pairs[1 * stride + 0] * y_pairs[0 * stride + 1] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = x_pairs[1 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = x_pairs[0 * stride + 1] * y_pairs[1 * stride + 0] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[1 * stride + 1] *
                 z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[1 * stride + 0] *
                 z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = x_pairs[0 * stride + 1] * y_pairs[0 * stride + 0] *
                 z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 1] *
                 z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = x_pairs[0 * stride + 0] * y_pairs[0 * stride + 0] *
                 z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 2, expression);
  }
  if constexpr (i_angular == 1 && j_angular == 2) {
    expression = 1.0925484305920792 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.31539156525252 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.5462742152960396 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.31539156525252 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 0.5462742152960396 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.31539156525252 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = 0.5462742152960396 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 4, expression);
  }
  if constexpr (i_angular == 1 && j_angular == 3) {
    expression = 1.7701307697799304 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -1.1195289977703462 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.1195289977703462 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.7463526651802308 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = -0.4570457994644657 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 1.4453057213202771 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.4453057213202771 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = 0.5900435899266435 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -1.1195289977703462 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = 1.4453057213202771 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = 0.5900435899266435 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[0 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -1.1195289977703462 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.4570457994644657 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = 1.4453057213202771 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = 0.5900435899266435 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 6, expression);
  }
  if constexpr (i_angular == 1 && j_angular == 4) {
    expression = 2.5033429417967046 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.7701307697799304 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.94617469575756 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.31735664074561293 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.31735664074561293 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.8462843753216345 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = -2.0071396306718676 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = -0.47308734787878 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.47308734787878 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 5.310392309339791 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 7, expression);
    expression = 0.6258357354491761 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 8, expression);
    expression = 2.5033429417967046 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.94617469575756 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 0.31735664074561293 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = -0.47308734787878 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 5.310392309339791 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 1 * n_functions + 7, expression);
    expression = 0.6258357354491761 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 1 * n_functions + 8, expression);
    expression = 2.5033429417967046 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.5033429417967046 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.94617469575756 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.94617469575756 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 5.6770481745453605 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = 0.31735664074561293 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6347132814912259 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -2.0071396306718676 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = -0.47308734787878 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.8385240872726802 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = 1.7701307697799304 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.310392309339791 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 2 * n_functions + 7, expression);
    expression = 0.6258357354491761 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 3.755014412695057 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 2 * n_functions + 8, expression);
  }
  if constexpr (i_angular == 2 && j_angular == 0) {
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = -0.31539156525252 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 0.5462742152960396 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
  }
  if constexpr (i_angular == 2 && j_angular == 1) {
    expression = 1.0925484305920792 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = 1.0925484305920792 * x_pairs[0 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -0.31539156525252 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.31539156525252 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.31539156525252 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.31539156525252 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.63078313050504 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 1.0925484305920792 * x_pairs[1 * stride + 0] *
                 y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 0.5462742152960396 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 0.5462742152960396 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = 0.5462742152960396 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.5462742152960396 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 2, expression);
  }
  if constexpr (i_angular == 2 && j_angular == 2) {
    expression = 1.1936620731892154 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.1936620731892154 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.34458055963862005 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.34458055963862005 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.6891611192772401 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 1.1936620731892154 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.5968310365946077 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5968310365946077 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 1.1936620731892154 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 1.1936620731892154 * x_pairs[0 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.34458055963862005 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.34458055963862005 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.6891611192772401 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = 1.1936620731892154 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 0.5968310365946077 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.5968310365946077 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -0.34458055963862005 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.34458055963862005 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.6891611192772401 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.34458055963862005 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.34458055963862005 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.6891611192772401 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.09947183943243458 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.09947183943243458 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.19894367886486916 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.09947183943243458 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.09947183943243458 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.19894367886486916 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.19894367886486916 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.19894367886486916 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.3978873577297383 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -0.34458055963862005 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.34458055963862005 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6891611192772401 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.17229027981931003 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.17229027981931003 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.17229027981931003 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.17229027981931003 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.34458055963862005 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.34458055963862005 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = 1.1936620731892154 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 1.1936620731892154 * x_pairs[1 * stride + 0] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = -0.34458055963862005 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.34458055963862005 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.6891611192772401 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 1.1936620731892154 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = 0.5968310365946077 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.5968310365946077 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = 0.5968310365946077 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5968310365946077 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 0.5968310365946077 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.5968310365946077 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.17229027981931003 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.17229027981931003 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.34458055963862005 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.17229027981931003 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.17229027981931003 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.34458055963862005 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = 0.5968310365946077 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.5968310365946077 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = 0.29841551829730384 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.29841551829730384 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.29841551829730384 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.29841551829730384 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 4, expression);
  }
  if constexpr (i_angular == 2 && j_angular == 3) {
    expression = 1.933953594465812 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.6446511981552707 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 3.1581329951084434 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.4993446709136042 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4993446709136042 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.9973786836544167 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -1.2231396495163152 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.2231396495163152 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.8154264330108767 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = -0.4993446709136042 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4993446709136042 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.9973786836544167 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 1.5790664975542217 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.5790664975542217 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = 0.6446511981552707 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.933953594465812 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 1.933953594465812 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.6446511981552707 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 3.1581329951084434 * x_pairs[0 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.4993446709136042 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.4993446709136042 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 1.9973786836544167 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -1.2231396495163152 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.2231396495163152 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 0.8154264330108767 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = -0.4993446709136042 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.4993446709136042 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.9973786836544167 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = 1.5790664975542217 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = 0.6446511981552707 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.933953594465812 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = -0.5582843141825403 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.18609477139418013 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.5582843141825403 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.18609477139418013 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.1165686283650806 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 0.37218954278836025 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.9116744674312494 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.9116744674312494 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.8233489348624987 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.1441483900851872 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.5765935603407488 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.1441483900851872 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.5765935603407488 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.2882967801703744 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 0.2882967801703744 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 1.1531871206814976 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 0.35309000295237447 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.35309000295237447 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.23539333530158296 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.35309000295237447 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.35309000295237447 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.23539333530158296 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 0.7061800059047489 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 0.7061800059047489 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 0.4707866706031659 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = 0.1441483900851872 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5765935603407488 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.1441483900851872 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5765935603407488 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2882967801703744 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.2882967801703744 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.1531871206814976 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -0.4558372337156247 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.4558372337156247 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.4558372337156247 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.4558372337156247 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.9116744674312494 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 0.9116744674312494 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = -0.18609477139418013 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.5582843141825403 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.18609477139418013 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.5582843141825403 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.37218954278836025 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.1165686283650806 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = 1.933953594465812 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.6446511981552707 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 3.1581329951084434 * x_pairs[1 * stride + 1] *
                 y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = -0.4993446709136042 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.4993446709136042 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 1.9973786836544167 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = -1.2231396495163152 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.2231396495163152 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 0.8154264330108767 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = -0.4993446709136042 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.4993446709136042 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.9973786836544167 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = 1.5790664975542217 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.5790664975542217 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = 0.6446511981552707 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.933953594465812 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = 0.966976797232906 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.32232559907763536 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.966976797232906 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.32232559907763536 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 1.5790664975542217 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.2496723354568021 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.2496723354568021 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 0.9986893418272084 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.2496723354568021 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.9986893418272084 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -0.6115698247581576 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.6115698247581576 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.40771321650543835 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.6115698247581576 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6115698247581576 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.40771321650543835 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = -0.2496723354568021 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2496723354568021 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.9986893418272084 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.2496723354568021 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.9986893418272084 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = 0.7895332487771108 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.7895332487771108 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.7895332487771108 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.7895332487771108 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = 0.32232559907763536 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.966976797232906 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.32232559907763536 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.966976797232906 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 6, expression);
  }
  if constexpr (i_angular == 2 && j_angular == 4) {
    expression = 2.735023402293748 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 5.801860783397436 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.933953594465812 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -1.0337416789158602 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.0337416789158602 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 6.202450073495162 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -2.1928972534697144 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.3467274997845937 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6934549995691874 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.7738199982767497 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.3467274997845937 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.7738199982767497 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.9246066660922498 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = -2.1928972534697144 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = -0.5168708394579301 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 3.101225036747581 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.5168708394579301 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 3.101225036747581 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 1.933953594465812 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 7, expression);
    expression = 0.683755850573437 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 4.102535103440622 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.683755850573437 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 8, expression);
    expression = 2.735023402293748 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.735023402293748 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 5.801860783397436 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.933953594465812 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -1.0337416789158602 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.0337416789158602 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 6.202450073495162 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -2.1928972534697144 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 0.3467274997845937 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6934549995691874 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.7738199982767497 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.3467274997845937 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 2.7738199982767497 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.9246066660922498 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -2.1928972534697144 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = -0.5168708394579301 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.101225036747581 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.5168708394579301 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 3.101225036747581 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = 1.933953594465812 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 7, expression);
    expression = 0.683755850573437 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 4.102535103440622 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.683755850573437 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 8, expression);
    expression = -0.7895332487771107 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.7895332487771107 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.7895332487771107 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.7895332487771107 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.5790664975542215 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.5790664975542215 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -1.674852942547621 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.5582843141825403 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.674852942547621 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.5582843141825403 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] +
                 3.349705885095242 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 1.1165686283650806 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.29841551829730373 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.29841551829730373 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.7904931097838226 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.29841551829730373 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.29841551829730373 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.7904931097838226 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.5968310365946075 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 0.5968310365946075 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 3.580986219567645 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 0.6330349097979652 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.6330349097979652 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 0.8440465463972869 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] +
                 0.6330349097979652 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.6330349097979652 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 0.8440465463972869 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3] -
                 1.2660698195959303 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 1.2660698195959303 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] +
                 1.6880930927945739 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.10009160766804052 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.20018321533608105 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.8007328613443242 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.10009160766804052 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.8007328613443242 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.2669109537814414 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.10009160766804052 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.20018321533608105 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.8007328613443242 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.10009160766804052 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.8007328613443242 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.2669109537814414 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.20018321533608105 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.4003664306721621 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.6014657226886484 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.20018321533608105 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 1.6014657226886484 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] +
                 0.5338219075628828 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 4];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = 0.6330349097979652 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6330349097979652 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.8440465463972869 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.6330349097979652 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6330349097979652 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.8440465463972869 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 1.2660698195959303 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 1.2660698195959303 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 1.6880930927945739 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = 0.14920775914865186 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.8952465548919113 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.14920775914865186 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.8952465548919113 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.14920775914865186 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.8952465548919113 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.14920775914865186 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.8952465548919113 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.29841551829730373 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.7904931097838226 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.29841551829730373 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 1.7904931097838226 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = -0.5582843141825403 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.674852942547621 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.5582843141825403 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.674852942547621 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.1165686283650806 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 3.349705885095242 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 7, expression);
    expression = -0.19738331219427768 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.184299873165666 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.19738331219427768 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.19738331219427768 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.184299873165666 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.19738331219427768 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.39476662438855536 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 2.368599746331332 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.39476662438855536 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 8, expression);
    expression = 2.735023402293748 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 5.801860783397436 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.933953594465812 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = -1.0337416789158602 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.0337416789158602 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 6.202450073495162 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = -2.1928972534697144 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = 0.3467274997845937 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6934549995691874 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.7738199982767497 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.3467274997845937 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 2.7738199982767497 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.9246066660922498 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = -2.1928972534697144 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = -0.5168708394579301 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.101225036747581 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.5168708394579301 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 3.101225036747581 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = 1.933953594465812 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 3 * n_functions + 7, expression);
    expression = 0.683755850573437 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 4.102535103440622 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.683755850573437 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 3 * n_functions + 8, expression);
    expression = 1.367511701146874 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.367511701146874 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.367511701146874 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.367511701146874 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 2.900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.966976797232906 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 2.900930391698718 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.966976797232906 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.5168708394579301 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5168708394579301 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.101225036747581 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.5168708394579301 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.5168708394579301 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 3.101225036747581 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -1.0964486267348572 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.0964486267348572 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.461931502313143 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] +
                 1.0964486267348572 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.0964486267348572 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.461931502313143 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = 0.17336374989229686 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.3467274997845937 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.3869099991383749 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.17336374989229686 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.3869099991383749 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.4623033330461249 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.17336374989229686 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.3467274997845937 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.3869099991383749 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.17336374989229686 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.3869099991383749 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.4623033330461249 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = -1.0964486267348572 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.0964486267348572 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.461931502313143 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 1.0964486267348572 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.0964486267348572 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.461931502313143 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = -0.25843541972896505 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.5506125183737904 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.25843541972896505 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5506125183737904 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.25843541972896505 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.5506125183737904 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.25843541972896505 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.5506125183737904 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 4 * n_functions + 6, expression);
    expression = 0.966976797232906 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.900930391698718 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.966976797232906 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.900930391698718 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 4 * n_functions + 7, expression);
    expression = 0.3418779252867185 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.051267551720311 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3418779252867185 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.3418779252867185 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.051267551720311 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.3418779252867185 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 4 * n_functions + 8, expression);
  }
  if constexpr (i_angular == 3 && j_angular == 0) {
    expression = 1.7701307697799304 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = -0.4570457994644657 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -1.1195289977703462 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -0.4570457994644657 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 1.4453057213202771 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = 0.5900435899266435 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
  }
  if constexpr (i_angular == 3 && j_angular == 1) {
    expression = 1.7701307697799304 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.7701307697799304 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = 1.7701307697799304 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.5900435899266435 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 2.8906114426405543 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 2.8906114426405543 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = 2.8906114426405543 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -0.4570457994644657 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.4570457994644657 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4570457994644657 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.8281831978578629 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -1.1195289977703462 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.1195289977703462 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.7463526651802308 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -1.1195289977703462 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = -1.1195289977703462 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.1195289977703462 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 0.7463526651802308 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = -0.4570457994644657 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = -0.4570457994644657 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4570457994644657 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.8281831978578629 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.4570457994644657 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4570457994644657 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.8281831978578629 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = 1.4453057213202771 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.4453057213202771 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = 1.4453057213202771 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = 1.4453057213202771 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.4453057213202771 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = 0.5900435899266435 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = 0.5900435899266435 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.7701307697799304 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = 0.5900435899266435 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.7701307697799304 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 2, expression);
  }
  if constexpr (i_angular == 3 && j_angular == 2) {
    expression = 1.933953594465812 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.6446511981552707 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 1.933953594465812 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.6446511981552707 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.5582843141825403 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5582843141825403 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1165686283650806 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.18609477139418013 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.18609477139418013 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.37218954278836025 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 1.933953594465812 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.6446511981552707 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.966976797232906 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.966976797232906 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.32232559907763536 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.32232559907763536 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 3.1581329951084434 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 3.1581329951084434 * x_pairs[1 * stride + 0] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -0.9116744674312494 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.9116744674312494 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.8233489348624987 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = 3.1581329951084434 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 1.5790664975542217 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.5790664975542217 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -0.4993446709136042 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4993446709136042 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.9973786836544167 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.4993446709136042 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.4993446709136042 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.9973786836544167 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.1441483900851872 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2882967801703744 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.1441483900851872 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2882967801703744 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.5765935603407488 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.5765935603407488 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.1531871206814976 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -0.4993446709136042 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4993446709136042 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.9973786836544167 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.2496723354568021 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2496723354568021 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.9986893418272084 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.9986893418272084 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -1.2231396495163152 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.2231396495163152 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.8154264330108767 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -1.2231396495163152 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.2231396495163152 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 0.8154264330108767 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 0.35309000295237447 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.35309000295237447 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.7061800059047489 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.35309000295237447 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.35309000295237447 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.7061800059047489 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.23539333530158296 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 0.23539333530158296 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 0.4707866706031659 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = -1.2231396495163152 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.2231396495163152 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 0.8154264330108767 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = -0.6115698247581576 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6115698247581576 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.6115698247581576 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6115698247581576 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.40771321650543835 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 0.40771321650543835 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = -0.4993446709136042 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4993446709136042 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.9973786836544167 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = -0.4993446709136042 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.4993446709136042 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.9973786836544167 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = 0.1441483900851872 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2882967801703744 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.1441483900851872 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.1441483900851872 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2882967801703744 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.5765935603407488 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.5765935603407488 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.1531871206814976 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -0.4993446709136042 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4993446709136042 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.9973786836544167 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = -0.2496723354568021 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2496723354568021 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2496723354568021 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.9986893418272084 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.9986893418272084 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = 1.5790664975542217 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.5790664975542217 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = 1.5790664975542217 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = -0.4558372337156247 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.4558372337156247 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.9116744674312494 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.4558372337156247 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.4558372337156247 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.9116744674312494 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = 1.5790664975542217 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = 0.7895332487771108 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.7895332487771108 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.7895332487771108 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.7895332487771108 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = 0.6446511981552707 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.933953594465812 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = 0.6446511981552707 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.933953594465812 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = -0.18609477139418013 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.18609477139418013 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.37218954278836025 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.5582843141825403 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.5582843141825403 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.1165686283650806 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = 0.6446511981552707 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.933953594465812 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = 0.32232559907763536 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.32232559907763536 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.966976797232906 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.966976797232906 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 4, expression);
  }
  if constexpr (i_angular == 3 && j_angular == 3) {
    expression = 3.133362942121689 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.044454314040563 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.044454314040563 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.34815143801352105 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 5.116760258096 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.7055867526986666 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.8090308328307184 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.8090308328307184 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.2361233313228737 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.26967694427690614 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.0787077771076246 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -1.981712726614177 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.981712726614177 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.3211418177427847 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.6605709088713924 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6605709088713924 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.4403806059142616 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = -0.8090308328307184 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.8090308328307184 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 3.2361233313228737 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.26967694427690614 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.0787077771076246 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 2.558380129048 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.558380129048 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.8527933763493333 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.8527933763493333 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = 1.044454314040563 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.133362942121689 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.34815143801352105 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.044454314040563 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 5.116760258096 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.7055867526986666 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 8.355634512324507 * x_pairs[1 * stride + 1] *
                 y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -1.3211418177427847 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.3211418177427847 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 5.284567270971139 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -3.236123331322874 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 3.236123331322874 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 2.1574155542152496 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = -1.3211418177427847 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.3211418177427847 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 5.284567270971139 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = 4.177817256162253 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 4.177817256162253 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = 1.7055867526986666 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.116760258096 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = -0.8090308328307184 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8090308328307184 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.2361233313228737 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.0787077771076246 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -1.3211418177427847 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.3211418177427847 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 5.284567270971139 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.20889086280811262 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.20889086280811262 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.8355634512324505 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 0.8355634512324505 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0] +
                 3.342253804929802 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 0.5116760258095999 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.5116760258095999 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.3411173505397333 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.5116760258095999 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.5116760258095999 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.3411173505397333 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3] -
                 2.0467041032383997 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 2.0467041032383997 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1] +
                 1.3644694021589332 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = 0.20889086280811262 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.20889086280811262 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.8355634512324505 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.8355634512324505 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 3.342253804929802 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -0.6605709088713924 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6605709088713924 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.6605709088713924 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6605709088713924 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.6422836354855694 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 2.6422836354855694 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = -0.26967694427690614 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.26967694427690614 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.0787077771076246 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 3.2361233313228737 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = -1.981712726614177 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.6605709088713924 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 1.981712726614177 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.6605709088713924 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] +
                 1.3211418177427847 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 0.4403806059142616 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -3.236123331322874 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 3.236123331322874 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.1574155542152496 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 0.5116760258095999 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.5116760258095999 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 2.0467041032383997 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 0.5116760258095999 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.5116760258095999 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 2.0467041032383997 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2] -
                 0.3411173505397333 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 0.3411173505397333 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0] +
                 1.3644694021589332 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 1.2533451768486759 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.2533451768486759 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 0.8355634512324506 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 1.2533451768486759 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.2533451768486759 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 0.8355634512324506 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3] -
                 0.8355634512324506 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 0.8355634512324506 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1] +
                 0.5570423008216338 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = 0.5116760258095999 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.5116760258095999 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.0467041032383997 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.5116760258095999 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.5116760258095999 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.0467041032383997 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.3411173505397333 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 0.3411173505397333 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 1.3644694021589332 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = -1.618061665661437 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.618061665661437 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.618061665661437 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.618061665661437 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] +
                 1.0787077771076248 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 1.0787077771076248 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = -0.6605709088713924 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.981712726614177 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.6605709088713924 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.981712726614177 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.4403806059142616 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 1.3211418177427847 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = -0.8090308328307184 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8090308328307184 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.26967694427690614 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.2361233313228737 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.0787077771076246 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = -1.3211418177427847 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.3211418177427847 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 5.284567270971139 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = 0.20889086280811262 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.20889086280811262 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.8355634512324505 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 0.8355634512324505 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 3.342253804929802 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = 0.5116760258095999 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.5116760258095999 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.3411173505397333 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.5116760258095999 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.5116760258095999 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.3411173505397333 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 2.0467041032383997 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 2.0467041032383997 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 1.3644694021589332 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = 0.20889086280811262 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.20889086280811262 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20889086280811262 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8355634512324505 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.8355634512324505 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.8355634512324505 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 3.342253804929802 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = -0.6605709088713924 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6605709088713924 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.6605709088713924 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6605709088713924 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 2.6422836354855694 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 2.6422836354855694 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = -0.26967694427690614 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.26967694427690614 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.0787077771076246 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 3.2361233313228737 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 6, expression);
    expression = 2.558380129048 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.8527933763493333 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 2.558380129048 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.8527933763493333 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = 4.177817256162253 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 4.177817256162253 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = -0.6605709088713924 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.6605709088713924 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 2.6422836354855694 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 0.6605709088713924 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.6605709088713924 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 2.6422836354855694 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = -1.618061665661437 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.618061665661437 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 1.0787077771076248 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 1.618061665661437 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.618061665661437 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.0787077771076248 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = -0.6605709088713924 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.6605709088713924 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 2.6422836354855694 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.6605709088713924 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6605709088713924 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.6422836354855694 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = 2.0889086280811267 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.0889086280811267 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.0889086280811267 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.0889086280811267 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 5, expression);
    expression = 0.8527933763493333 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.558380129048 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.8527933763493333 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.558380129048 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 6, expression);
    expression = 1.044454314040563 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.34815143801352105 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 3.133362942121689 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.044454314040563 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = 1.7055867526986666 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 5.116760258096 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = -0.26967694427690614 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.26967694427690614 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.0787077771076246 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.8090308328307184 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 3.2361233313228737 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = -0.6605709088713924 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.6605709088713924 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.4403806059142616 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 1.981712726614177 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.981712726614177 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.3211418177427847 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = -0.26967694427690614 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.26967694427690614 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.0787077771076246 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.8090308328307184 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8090308328307184 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 3.2361233313228737 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 6 * n_functions + 4, expression);
    expression = 0.8527933763493333 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.8527933763493333 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 2.558380129048 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.558380129048 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 5, expression);
    expression = 0.34815143801352105 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.044454314040563 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.044454314040563 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 3.133362942121689 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 6, expression);
  }
  if constexpr (i_angular == 3 && j_angular == 4) {
    expression = 4.431244368585756 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 4.431244368585756 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.477081456195252 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.477081456195252 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 9.400088826365067 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 3.133362942121689 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 3.1333629421216895 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.044454314040563 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -1.674852942547621 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.674852942547621 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 10.049117655285729 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.5582843141825404 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.5582843141825404 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 3.349705885095243 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -3.552899619496998 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 3.552899619496998 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] +
                 4.737199492662665 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3] +
                 1.1842998731656662 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.1842998731656662 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.5617627547778047 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.1235255095556094 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 4.4941020382224375 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.5617627547778047 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 4.4941020382224375 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.4980340127408125 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.18725425159260156 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.3745085031852031 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.4980340127408125 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.18725425159260156 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.4993446709136042 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = -3.552899619496998 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.552899619496998 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 4.737199492662665 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 1.1842998731656662 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.1842998731656662 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.5790664975542217 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = -0.8374264712738105 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 5.024558827642864 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.8374264712738105 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 5.024558827642864 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.2791421570912702 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.6748529425476215 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2791421570912702 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.6748529425476215 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 3.133362942121689 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 9.400088826365067 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.044454314040563 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 3.1333629421216895 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 7, expression);
    expression = 1.107811092146439 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 6.646866552878635 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.107811092146439 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.369270364048813 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.2156221842928785 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.369270364048813 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 8, expression);
    expression = 7.236191752411021 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 7.236191752411021 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 15.350280774287999 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.116760258096 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -2.735023402293748 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 16.41014041376249 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -5.801860783397436 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] +
                 7.735814377863249 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 0.9173547371372363 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.8347094742744725 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 7.33883789709789 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.9173547371372363 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 7.33883789709789 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] +
                 2.4462792990326303 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -5.801860783397436 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 7.735814377863249 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = -1.367511701146874 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 8.205070206881246 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 1.367511701146874 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 8.205070206881246 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = 5.116760258096 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 15.350280774287999 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 7, expression);
    expression = 1.8090479381027553 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 10.854287628616532 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.8090479381027553 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 8, expression);
    expression = -1.1441423761672023 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.1441423761672023 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] +
                 4.576569504668809 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 4.576569504668809 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -2.427092498492155 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.8090308328307184 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 2.427092498492155 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.8090308328307184 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] +
                 9.70836999396862 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1] -
                 3.2361233313228737 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.4324451702555616 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.59467102153337 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.4324451702555616 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.59467102153337 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2] -
                 1.7297806810222465 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.7297806810222465 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0] +
                 10.37868408613348 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 0.9173547371372361 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3] +
                 0.9173547371372361 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 3] -
                 3.6694189485489446 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1] -
                 3.6694189485489446 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 1] +
                 4.89255859806526 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.1450465195849359 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.1450465195849359 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.3867907188931624 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.1450465195849359 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.1450465195849359 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.3867907188931624 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.5801860783397436 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.5801860783397436 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 2] +
                 1.5471628755726496 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 4];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = 0.9173547371372361 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.9173547371372361 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3] -
                 3.6694189485489446 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 3.6694189485489446 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1] +
                 4.89255859806526 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = 0.2162225851277808 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.297335510766685 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2162225851277808 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.297335510766685 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.2162225851277808 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.297335510766685 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2162225851277808 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.297335510766685 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.8648903405111232 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] +
                 5.18934204306674 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.8648903405111232 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0] -
                 5.18934204306674 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = -0.8090308328307184 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.427092498492155 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.8090308328307184 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.427092498492155 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] +
                 3.2361233313228737 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 9.70836999396862 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 7, expression);
    expression = -0.28603559404180057 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 6.864854257003214 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.1441423761672023 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 8, expression);
    expression = -2.802565014705135 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.802565014705135 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 2.802565014705135 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.802565014705135 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] +
                 1.8683766764700898 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 1.8683766764700898 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -5.945138179842531 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] +
                 1.981712726614177 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 5.945138179842531 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 1.981712726614177 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] +
                 3.963425453228354 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1] -
                 1.3211418177427847 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 1.0592700088571234 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.0592700088571234 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 6.355620053142741 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 1.0592700088571234 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.0592700088571234 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 6.355620053142741 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2] -
                 0.7061800059047489 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 0.7061800059047489 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0] +
                 4.237080035428494 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 2.2470510191112187 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 2.996068025481625 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3] +
                 2.2470510191112187 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] -
                 2.996068025481625 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 3] -
                 1.4980340127408125 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1] -
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 1] +
                 1.9973786836544167 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = -0.35528996194969986 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.7105799238993997 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 2.842319695597599 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.35528996194969986 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] +
                 2.842319695597599 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] -
                 0.947439898532533 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4] -
                 0.35528996194969986 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.7105799238993997 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 2.842319695597599 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.35528996194969986 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 2.842319695597599 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 0.947439898532533 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 4] +
                 0.23685997463313324 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] +
                 0.4737199492662665 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] -
                 1.894879797065066 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2] +
                 0.23685997463313324 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0] -
                 1.894879797065066 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 2] +
                 0.6316265990216886 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 4];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = 2.2470510191112187 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.996068025481625 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 2.2470510191112187 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.996068025481625 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3] -
                 1.4980340127408125 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 1.4980340127408125 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1] +
                 1.9973786836544167 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = 0.5296350044285617 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 3.1778100265713705 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.5296350044285617 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] +
                 3.1778100265713705 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.5296350044285617 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 3.1778100265713705 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.5296350044285617 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 3.1778100265713705 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 0.35309000295237447 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] +
                 2.118540017714247 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2] +
                 0.35309000295237447 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0] -
                 2.118540017714247 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = -1.981712726614177 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 5.945138179842531 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.981712726614177 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 5.945138179842531 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] +
                 1.3211418177427847 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 3.963425453228354 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 7, expression);
    expression = -0.7006412536762837 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 4.2038475220577025 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.7006412536762837 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 0.7006412536762837 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 4.2038475220577025 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.7006412536762837 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 0.46709416911752244 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 2.802565014705135 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 0.46709416911752244 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 8, expression);
    expression = -1.1441423761672023 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.1441423761672023 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 4.576569504668809 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 4.576569504668809 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = -2.427092498492155 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.8090308328307184 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 2.427092498492155 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.8090308328307184 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] +
                 9.70836999396862 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 3.2361233313228737 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = 0.4324451702555616 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.59467102153337 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.4324451702555616 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.59467102153337 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 1.7297806810222465 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.7297806810222465 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 10.37868408613348 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = 0.9173547371372361 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] +
                 0.9173547371372361 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3] -
                 3.6694189485489446 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 3.6694189485489446 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] +
                 4.89255859806526 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = -0.1450465195849359 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.1450465195849359 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.3867907188931624 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.1450465195849359 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.1450465195849359 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.3867907188931624 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.5801860783397436 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.5801860783397436 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] +
                 1.5471628755726496 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 4];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = 0.9173547371372361 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 0.9173547371372361 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.9173547371372361 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.223139649516315 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 3.6694189485489446 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 3.6694189485489446 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 4.89255859806526 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = 0.2162225851277808 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.297335510766685 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2162225851277808 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.297335510766685 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.2162225851277808 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.297335510766685 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2162225851277808 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.297335510766685 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.8648903405111232 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 5.18934204306674 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.8648903405111232 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 5.18934204306674 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2];
    atomicAdd(output + 4 * n_functions + 6, expression);
    expression = -0.8090308328307184 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.427092498492155 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.8090308328307184 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.427092498492155 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 3.2361233313228737 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 9.70836999396862 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 4 * n_functions + 7, expression);
    expression = -0.28603559404180057 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 6.864854257003214 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.1441423761672023 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0];
    atomicAdd(output + 4 * n_functions + 8, expression);
    expression = 3.6180958762055107 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 3.6180958762055107 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.6180958762055107 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 3.6180958762055107 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = 7.675140387143999 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.558380129048 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 7.675140387143999 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.558380129048 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = -1.367511701146874 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.367511701146874 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 8.205070206881246 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 1.367511701146874 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.367511701146874 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 8.205070206881246 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = -2.900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.900930391698718 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] +
                 3.8679071889316243 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3] +
                 2.900930391698718 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] -
                 3.8679071889316243 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = 0.4586773685686181 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.9173547371372363 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.669418948548945 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.4586773685686181 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 3.669418948548945 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 1.2231396495163152 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4] -
                 0.4586773685686181 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.9173547371372363 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 3.669418948548945 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.4586773685686181 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 3.669418948548945 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.2231396495163152 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = -2.900930391698718 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.900930391698718 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 3.8679071889316243 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 2.900930391698718 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 3.8679071889316243 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 5 * n_functions + 5, expression);
    expression = -0.683755850573437 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 4.102535103440623 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.683755850573437 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 4.102535103440623 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.683755850573437 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 4.102535103440623 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.683755850573437 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 4.102535103440623 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 5 * n_functions + 6, expression);
    expression = 2.558380129048 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 7.675140387143999 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.558380129048 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 7.675140387143999 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 5 * n_functions + 7, expression);
    expression = 0.9045239690513777 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.427143814308266 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 0.9045239690513777 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 0.9045239690513777 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 5.427143814308266 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.9045239690513777 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 5 * n_functions + 8, expression);
    expression = 1.477081456195252 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.477081456195252 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 4.431244368585756 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 4.431244368585756 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = 3.1333629421216895 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.044454314040563 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 9.400088826365067 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 3.133362942121689 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = -0.5582843141825404 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.5582843141825404 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.349705885095243 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.674852942547621 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.674852942547621 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 10.049117655285729 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = -1.1842998731656662 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.1842998731656662 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.5790664975542217 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] +
                 3.552899619496998 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 3.552899619496998 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 4.737199492662665 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = 0.18725425159260156 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.3745085031852031 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.4980340127408125 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.18725425159260156 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.4980340127408125 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.4993446709136042 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.5617627547778047 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.1235255095556094 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 4.4941020382224375 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.5617627547778047 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 4.4941020382224375 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 1.4980340127408125 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 6 * n_functions + 4, expression);
    expression = -1.1842998731656662 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.1842998731656662 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.5790664975542217 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 3.552899619496998 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 3.552899619496998 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 4.737199492662665 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 6 * n_functions + 5, expression);
    expression = -0.2791421570912702 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.6748529425476215 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.2791421570912702 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.6748529425476215 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.8374264712738105 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 5.024558827642864 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.8374264712738105 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 5.024558827642864 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 6 * n_functions + 6, expression);
    expression = 1.044454314040563 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.1333629421216895 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 3.133362942121689 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 9.400088826365067 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 6 * n_functions + 7, expression);
    expression = 0.369270364048813 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.2156221842928785 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.369270364048813 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.107811092146439 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 6.646866552878635 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.107811092146439 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 6 * n_functions + 8, expression);
  }
  if constexpr (i_angular == 4 && j_angular == 0) {
    expression = 2.5033429417967046 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = -0.94617469575756 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -2.0071396306718676 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = 0.31735664074561293 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = -2.0071396306718676 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = -0.47308734787878 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = 1.7701307697799304 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.310392309339791 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 0, expression);
    expression = 0.6258357354491761 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 0, expression);
  }
  if constexpr (i_angular == 4 && j_angular == 1) {
    expression = 2.5033429417967046 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 2.5033429417967046 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5033429417967046 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = 2.5033429417967046 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.5033429417967046 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 5.310392309339791 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 5.310392309339791 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = 5.310392309339791 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.7701307697799304 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -0.94617469575756 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -0.94617469575756 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.94617469575756 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 5.6770481745453605 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = -0.94617469575756 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.94617469575756 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 5.6770481745453605 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -2.0071396306718676 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = -2.0071396306718676 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.0071396306718676 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.676186174229157 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 0.31735664074561293 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.31735664074561293 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.8462843753216345 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 0.31735664074561293 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.6347132814912259 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = 0.31735664074561293 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6347132814912259 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.5388531259649034 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.31735664074561293 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.5388531259649034 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.8462843753216345 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -2.0071396306718676 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = -2.0071396306718676 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.0071396306718676 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.676186174229157 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = -2.0071396306718676 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.0071396306718676 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.676186174229157 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = -0.47308734787878 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.47308734787878 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = -0.47308734787878 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 2.8385240872726802 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = -0.47308734787878 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.8385240872726802 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.47308734787878 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.8385240872726802 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = 1.7701307697799304 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.310392309339791 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 0, expression);
    expression = 1.7701307697799304 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 5.310392309339791 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 1, expression);
    expression = 1.7701307697799304 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.310392309339791 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 2, expression);
    expression = 0.6258357354491761 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 0, expression);
    expression = 0.6258357354491761 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 3.755014412695057 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 1, expression);
    expression = 0.6258357354491761 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.755014412695057 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6258357354491761 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 2, expression);
  }
  if constexpr (i_angular == 4 && j_angular == 2) {
    expression = 2.735023402293748 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 2.735023402293748 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.735023402293748 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -0.7895332487771107 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.7895332487771107 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.5790664975542215 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.7895332487771107 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.7895332487771107 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.5790664975542215 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = 2.735023402293748 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 1.367511701146874 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.367511701146874 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.367511701146874 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.367511701146874 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 5.801860783397436 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.933953594465812 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 5.801860783397436 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 1.933953594465812 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -1.674852942547621 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.674852942547621 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 3.349705885095242 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.5582843141825403 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.5582843141825403 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.1165686283650806 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = 5.801860783397436 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.933953594465812 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 2.900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.900930391698718 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 0.966976797232906 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.966976797232906 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -1.0337416789158602 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.0337416789158602 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 6.202450073495162 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -1.0337416789158602 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.0337416789158602 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 6.202450073495162 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.29841551829730373 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.29841551829730373 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5968310365946075 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.29841551829730373 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.29841551829730373 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5968310365946075 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 1.7904931097838226 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.7904931097838226 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 3.580986219567645 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = -1.0337416789158602 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.0337416789158602 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 6.202450073495162 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.5168708394579301 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.5168708394579301 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5168708394579301 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.5168708394579301 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 3.101225036747581 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 3.101225036747581 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -2.1928972534697144 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.1928972534697144 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.923863004626286 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -2.1928972534697144 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 0.6330349097979652 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6330349097979652 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.2660698195959303 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.6330349097979652 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6330349097979652 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.2660698195959303 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.8440465463972869 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] -
                 0.8440465463972869 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0] +
                 1.6880930927945739 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = -2.1928972534697144 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = -1.0964486267348572 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.0964486267348572 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.0964486267348572 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.0964486267348572 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.461931502313143 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] -
                 1.461931502313143 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = 0.3467274997845937 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.6934549995691874 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.7738199982767497 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.3467274997845937 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.7738199982767497 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.9246066660922498 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 0.3467274997845937 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.6934549995691874 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.7738199982767497 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 0.3467274997845937 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.7738199982767497 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 0.9246066660922498 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.10009160766804052 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.10009160766804052 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.20018321533608105 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.20018321533608105 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.20018321533608105 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.4003664306721621 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.8007328613443242 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.8007328613443242 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.6014657226886484 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.10009160766804052 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.10009160766804052 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.20018321533608105 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.8007328613443242 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.8007328613443242 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.6014657226886484 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.2669109537814414 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] -
                 0.2669109537814414 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0] +
                 0.5338219075628828 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = 0.3467274997845937 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.6934549995691874 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.7738199982767497 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.3467274997845937 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.7738199982767497 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.9246066660922498 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = 0.17336374989229686 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.17336374989229686 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3467274997845937 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.3467274997845937 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.3869099991383749 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.3869099991383749 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.17336374989229686 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.17336374989229686 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.3869099991383749 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.3869099991383749 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.4623033330461249 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] -
                 0.4623033330461249 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = -2.1928972534697144 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.1928972534697144 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.923863004626286 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = -2.1928972534697144 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = 0.6330349097979652 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6330349097979652 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.2660698195959303 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.6330349097979652 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.6330349097979652 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.2660698195959303 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.8440465463972869 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 0.8440465463972869 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 1.6880930927945739 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = -2.1928972534697144 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.1928972534697144 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.923863004626286 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = -1.0964486267348572 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.0964486267348572 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.0964486267348572 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.0964486267348572 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.461931502313143 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 1.461931502313143 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = -0.5168708394579301 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 3.101225036747581 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 0.5168708394579301 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 3.101225036747581 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = -0.5168708394579301 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 3.101225036747581 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 0.5168708394579301 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 3.101225036747581 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = 0.14920775914865186 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.14920775914865186 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.29841551829730373 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.8952465548919113 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 0.8952465548919113 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.7904931097838226 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.14920775914865186 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.14920775914865186 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.29841551829730373 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.8952465548919113 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 0.8952465548919113 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.7904931097838226 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = -0.5168708394579301 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 3.101225036747581 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 0.5168708394579301 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.101225036747581 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = -0.25843541972896505 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.25843541972896505 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.5506125183737904 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.5506125183737904 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.25843541972896505 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.25843541972896505 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.5506125183737904 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.5506125183737904 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 4, expression);
    expression = 1.933953594465812 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 0, expression);
    expression = 1.933953594465812 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 1, expression);
    expression = -0.5582843141825403 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.5582843141825403 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.1165686283650806 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 1.674852942547621 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.674852942547621 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.349705885095242 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 7 * n_functions + 2, expression);
    expression = 1.933953594465812 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 3, expression);
    expression = 0.966976797232906 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.966976797232906 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 2.900930391698718 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.900930391698718 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 4, expression);
    expression = 0.683755850573437 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 4.102535103440622 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.683755850573437 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 0, expression);
    expression = 0.683755850573437 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 4.102535103440622 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.683755850573437 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 1, expression);
    expression = -0.19738331219427768 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.19738331219427768 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.39476662438855536 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.184299873165666 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.184299873165666 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.368599746331332 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.19738331219427768 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.19738331219427768 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.39476662438855536 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 8 * n_functions + 2, expression);
    expression = 0.683755850573437 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 4.102535103440622 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.683755850573437 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 3, expression);
    expression = 0.3418779252867185 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.3418779252867185 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.051267551720311 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 2.051267551720311 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3418779252867185 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.3418779252867185 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 4, expression);
  }
  if constexpr (i_angular == 4 && j_angular == 3) {
    expression = 4.431244368585756 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.477081456195252 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 4.431244368585756 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.477081456195252 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 7.236191752411021 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 7.236191752411021 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -1.1441423761672023 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.1441423761672023 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 4.576569504668809 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.1441423761672023 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 4.576569504668809 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -2.802565014705135 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.802565014705135 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.8683766764700898 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 2.802565014705135 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.802565014705135 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.8683766764700898 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = -1.1441423761672023 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.1441423761672023 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 4.576569504668809 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.1441423761672023 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 4.576569504668809 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = 3.6180958762055107 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.6180958762055107 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 3.6180958762055107 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 3.6180958762055107 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = 1.477081456195252 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 4.431244368585756 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.477081456195252 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 4.431244368585756 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 9.400088826365067 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 3.1333629421216895 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.133362942121689 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.044454314040563 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 15.350280774287999 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.116760258096 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -2.427092498492155 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 2.427092498492155 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 9.70836999396862 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2] +
                 0.8090308328307184 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.8090308328307184 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.2361233313228737 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -5.945138179842531 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 5.945138179842531 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 3.963425453228354 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3] +
                 1.981712726614177 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 1.981712726614177 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.3211418177427847 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = -2.427092498492155 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 2.427092498492155 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 9.70836999396862 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.8090308328307184 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.8090308328307184 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.2361233313228737 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = 7.675140387143999 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 7.675140387143999 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.558380129048 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.558380129048 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = 3.1333629421216895 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 9.400088826365067 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.044454314040563 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.133362942121689 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = -1.674852942547621 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.5582843141825404 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.674852942547621 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.5582843141825404 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] +
                 10.049117655285729 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 3.349705885095243 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -2.735023402293748 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 2.735023402293748 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 16.41014041376249 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.4324451702555616 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.7297806810222465 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.4324451702555616 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 1.7297806810222465 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2] -
                 2.59467102153337 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 2.59467102153337 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0] +
                 10.37868408613348 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 1.0592700088571234 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.0592700088571234 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.7061800059047489 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 1.0592700088571234 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.0592700088571234 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.7061800059047489 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3] -
                 6.355620053142741 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 6.355620053142741 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1] +
                 4.237080035428494 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = 0.4324451702555616 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.7297806810222465 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.4324451702555616 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.4324451702555616 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.7297806810222465 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 2.59467102153337 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 2.59467102153337 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 10.37868408613348 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = -1.367511701146874 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.367511701146874 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.367511701146874 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.367511701146874 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] +
                 8.205070206881246 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 8.205070206881246 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = -0.5582843141825404 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.674852942547621 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.5582843141825404 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.674852942547621 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 3.349705885095243 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 10.049117655285729 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = -3.552899619496998 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.1842998731656662 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.552899619496998 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.1842998731656662 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] +
                 4.737199492662665 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0] -
                 1.5790664975542217 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -5.801860783397436 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 7.735814377863249 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 0.9173547371372361 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2] +
                 0.9173547371372361 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 2] -
                 1.223139649516315 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0] -
                 1.223139649516315 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 0] +
                 4.89255859806526 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 2.2470510191112187 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.4980340127408125 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3] +
                 2.2470510191112187 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 3] -
                 2.996068025481625 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1] -
                 2.996068025481625 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 1] +
                 1.9973786836544167 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = 0.9173547371372361 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.9173547371372361 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 1.223139649516315 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] -
                 1.223139649516315 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0] +
                 4.89255859806526 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = -2.900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.900930391698718 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] +
                 3.8679071889316243 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1] -
                 3.8679071889316243 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = -1.1842998731656662 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.552899619496998 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.1842998731656662 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.552899619496998 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.5790664975542217 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] -
                 4.737199492662665 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = 0.5617627547778047 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.18725425159260156 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.1235255095556094 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.3745085031852031 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 4.4941020382224375 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.4980340127408125 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 0.5617627547778047 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.18725425159260156 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] -
                 4.4941020382224375 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] +
                 1.4980340127408125 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0] -
                 0.4993446709136042 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 0.9173547371372363 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.8347094742744725 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] -
                 7.33883789709789 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 0.9173547371372363 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 7.33883789709789 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 2.4462792990326303 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.1450465195849359 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.1450465195849359 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 0.5801860783397436 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.2900930391698718 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.1603721566794871 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.1450465195849359 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.1450465195849359 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 0.5801860783397436 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.1603721566794871 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.3867907188931624 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0] -
                 0.3867907188931624 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 0] +
                 1.5471628755726496 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -0.35528996194969986 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.35528996194969986 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.23685997463313324 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] -
                 0.7105799238993997 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.7105799238993997 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.4737199492662665 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] +
                 2.842319695597599 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 2.842319695597599 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] -
                 1.894879797065066 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3] -
                 0.35528996194969986 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.35528996194969986 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.23685997463313324 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3] +
                 2.842319695597599 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 2.842319695597599 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] -
                 1.894879797065066 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 3] -
                 0.947439898532533 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1] -
                 0.947439898532533 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 1] +
                 0.6316265990216886 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = -0.1450465195849359 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.1450465195849359 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.5801860783397436 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2900930391698718 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2900930391698718 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.1603721566794871 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.1450465195849359 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.1450465195849359 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.5801860783397436 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.1603721566794871 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.1603721566794871 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 4.6414886267179485 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.3867907188931624 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] -
                 0.3867907188931624 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0] +
                 1.5471628755726496 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 2];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = 0.4586773685686181 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4586773685686181 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.9173547371372363 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.9173547371372363 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 3.669418948548945 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 3.669418948548945 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 0.4586773685686181 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.4586773685686181 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] -
                 3.669418948548945 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 3.669418948548945 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] +
                 1.2231396495163152 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1] -
                 1.2231396495163152 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = 0.18725425159260156 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5617627547778047 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3745085031852031 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.1235255095556094 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.4980340127408125 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 4.4941020382224375 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.18725425159260156 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.5617627547778047 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.4980340127408125 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 4.4941020382224375 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.4993446709136042 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] -
                 1.4980340127408125 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 6, expression);
    expression = -3.552899619496998 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.1842998731656662 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.552899619496998 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.1842998731656662 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] +
                 4.737199492662665 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 1.5790664975542217 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = -5.801860783397436 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 5.801860783397436 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 7.735814377863249 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = 0.9173547371372361 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 0.9173547371372361 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2] -
                 1.223139649516315 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 1.223139649516315 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0] +
                 4.89255859806526 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = 2.2470510191112187 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.4980340127408125 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 2.2470510191112187 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.2470510191112187 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 1.4980340127408125 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3] -
                 2.996068025481625 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 2.996068025481625 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1] +
                 1.9973786836544167 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = 0.9173547371372361 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.9173547371372361 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 0.9173547371372361 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.6694189485489446 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 1.223139649516315 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 1.223139649516315 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 4.89255859806526 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = -2.900930391698718 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 2.900930391698718 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 2.900930391698718 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] +
                 3.8679071889316243 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 3.8679071889316243 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 5, expression);
    expression = -1.1842998731656662 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.552899619496998 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.1842998731656662 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.552899619496998 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.5790664975542217 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 4.737199492662665 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 6, expression);
    expression = -0.8374264712738105 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.2791421570912702 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 5.024558827642864 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.6748529425476215 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 0.8374264712738105 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.2791421570912702 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] -
                 5.024558827642864 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.6748529425476215 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = -1.367511701146874 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 8.205070206881246 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 1.367511701146874 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 8.205070206881246 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = 0.2162225851277808 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.2162225851277808 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 0.8648903405111232 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] -
                 1.297335510766685 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 1.297335510766685 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 5.18934204306674 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.2162225851277808 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.2162225851277808 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 0.8648903405111232 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.297335510766685 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 1.297335510766685 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] -
                 5.18934204306674 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = 0.5296350044285617 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.5296350044285617 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 0.35309000295237447 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] -
                 3.1778100265713705 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 3.1778100265713705 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 2.118540017714247 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3] -
                 0.5296350044285617 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.5296350044285617 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.35309000295237447 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3] +
                 3.1778100265713705 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 3.1778100265713705 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] -
                 2.118540017714247 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = 0.2162225851277808 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.2162225851277808 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8648903405111232 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 1.297335510766685 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.297335510766685 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 5.18934204306674 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.2162225851277808 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.2162225851277808 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.8648903405111232 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.297335510766685 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.297335510766685 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 5.18934204306674 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2];
    atomicAdd(output + 6 * n_functions + 4, expression);
    expression = -0.683755850573437 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.683755850573437 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 4.102535103440623 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 4.102535103440623 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 0.683755850573437 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.683755850573437 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] -
                 4.102535103440623 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 4.102535103440623 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 5, expression);
    expression = -0.2791421570912702 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.8374264712738105 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.6748529425476215 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 5.024558827642864 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 0.2791421570912702 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.8374264712738105 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.6748529425476215 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 5.024558827642864 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 6, expression);
    expression = 3.133362942121689 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.044454314040563 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 9.400088826365067 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 3.1333629421216895 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 0, expression);
    expression = 5.116760258096 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 15.350280774287999 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 1, expression);
    expression = -0.8090308328307184 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 0.8090308328307184 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 3.2361233313228737 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 2.427092498492155 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 2.427092498492155 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 9.70836999396862 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 7 * n_functions + 2, expression);
    expression = -1.981712726614177 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 1.981712726614177 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 1.3211418177427847 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 5.945138179842531 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 5.945138179842531 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 3.963425453228354 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 7 * n_functions + 3, expression);
    expression = -0.8090308328307184 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 0.8090308328307184 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 3.2361233313228737 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 2.427092498492155 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 2.427092498492155 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 9.70836999396862 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2];
    atomicAdd(output + 7 * n_functions + 4, expression);
    expression = 2.558380129048 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 2.558380129048 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 7.675140387143999 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 7.675140387143999 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 5, expression);
    expression = 1.044454314040563 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 3.133362942121689 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.1333629421216895 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 9.400088826365067 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 6, expression);
    expression = 1.107811092146439 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.369270364048813 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 6.646866552878635 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 2.2156221842928785 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.107811092146439 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.369270364048813 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 0, expression);
    expression = 1.8090479381027553 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 10.854287628616532 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.8090479381027553 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 1, expression);
    expression = -0.28603559404180057 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 1.7162135642508034 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 6.864854257003214 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.28603559404180057 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 8 * n_functions + 2, expression);
    expression = -0.7006412536762837 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.7006412536762837 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.46709416911752244 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 4.2038475220577025 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 4.2038475220577025 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 2.802565014705135 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 0.7006412536762837 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.7006412536762837 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.46709416911752244 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 8 * n_functions + 3, expression);
    expression = -0.28603559404180057 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.7162135642508034 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7162135642508034 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 6.864854257003214 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.28603559404180057 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.28603559404180057 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.1441423761672023 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2];
    atomicAdd(output + 8 * n_functions + 4, expression);
    expression = 0.9045239690513777 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.9045239690513777 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 5.427143814308266 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 5.427143814308266 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.9045239690513777 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.9045239690513777 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 5, expression);
    expression = 0.369270364048813 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.107811092146439 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.2156221842928785 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 6.646866552878635 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.369270364048813 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.107811092146439 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 6, expression);
  }
  if constexpr (i_angular == 4 && j_angular == 4) {
    expression = 6.266725884243379 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 6.266725884243379 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 6.266725884243379 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 6.266725884243379 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 0, expression);
    expression = 13.29373310575727 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 4.431244368585756 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 13.29373310575727 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 4.431244368585756 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 1, expression);
    expression = -2.368599746331332 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] -
                 2.368599746331332 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] +
                 14.211598477987994 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 2.368599746331332 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 2.368599746331332 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 14.211598477987994 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 2, expression);
    expression = -5.024558827642864 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] -
                 5.024558827642864 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] +
                 6.699411770190486 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3] +
                 5.024558827642864 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 5.024558827642864 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] -
                 6.699411770190486 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 3, expression);
    expression = 0.7944525066428426 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.5889050132856852 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 6.355620053142741 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.7944525066428426 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 6.355620053142741 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 2.118540017714247 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.7944525066428426 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 6.355620053142741 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.7944525066428426 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 6.355620053142741 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 2.118540017714247 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 0 * n_functions + 4, expression);
    expression = -5.024558827642864 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 5.024558827642864 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] +
                 6.699411770190486 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 5.024558827642864 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 5.024558827642864 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 6.699411770190486 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 0 * n_functions + 5, expression);
    expression = -1.184299873165666 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 7.105799238993997 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] +
                 1.184299873165666 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 7.105799238993997 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.184299873165666 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 7.105799238993997 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 1.184299873165666 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 7.105799238993997 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 0 * n_functions + 6, expression);
    expression = 4.431244368585756 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] -
                 13.29373310575727 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 4.431244368585756 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 13.29373310575727 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 0 * n_functions + 7, expression);
    expression = 1.5666814710608448 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 9.400088826365069 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.5666814710608448 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5666814710608448 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 9.400088826365069 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.5666814710608448 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 0 * n_functions + 8, expression);
    expression = 13.29373310575727 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 13.29373310575727 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 4.431244368585756 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 4.431244368585756 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 0, expression);
    expression = 28.200266479095202 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 9.400088826365067 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] -
                 9.400088826365067 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.133362942121689 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 1, expression);
    expression = -5.024558827642863 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] -
                 5.024558827642863 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] +
                 30.147352965857184 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2] +
                 1.674852942547621 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.674852942547621 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] -
                 10.049117655285729 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 2, expression);
    expression = -10.658698858490995 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] -
                 10.658698858490995 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] +
                 14.211598477987993 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 3] +
                 3.552899619496998 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 1] -
                 4.737199492662665 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 3, expression);
    expression = 1.685288264333414 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 3.370576528666828 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 13.482306114667312 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 1.685288264333414 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 13.482306114667312 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] +
                 4.4941020382224375 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 4] -
                 0.5617627547778047 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.1235255095556094 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] +
                 4.4941020382224375 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.5617627547778047 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0] +
                 4.4941020382224375 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 1 * n_functions + 4, expression);
    expression = -10.658698858490995 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 10.658698858490995 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] +
                 14.211598477987993 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3] +
                 3.552899619496998 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] -
                 4.737199492662665 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 1 * n_functions + 5, expression);
    expression = -2.5122794138214317 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 15.073676482928592 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] +
                 2.5122794138214317 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 15.073676482928592 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.8374264712738105 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.024558827642864 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.8374264712738105 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 1 * n_functions + 6, expression);
    expression = 9.400088826365067 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] -
                 28.200266479095202 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 3.133362942121689 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 9.400088826365067 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 1 * n_functions + 7, expression);
    expression = 3.3234332764393173 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 19.940599658635904 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 3.3234332764393173 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 1.107811092146439 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 6.646866552878635 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.107811092146439 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 1 * n_functions + 8, expression);
    expression = -2.368599746331332 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 2.368599746331332 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.368599746331332 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 2.368599746331332 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] +
                 14.211598477987994 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 14.211598477987994 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 0, expression);
    expression = -5.024558827642863 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.674852942547621 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 5.024558827642863 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.674852942547621 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] +
                 30.147352965857184 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1] -
                 10.049117655285729 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 1, expression);
    expression = 0.8952465548919112 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.8952465548919112 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 0] -
                 5.371479329351468 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 2] +
                 0.8952465548919112 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.8952465548919112 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 0] -
                 5.371479329351468 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 2] -
                 5.371479329351468 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 0] -
                 5.371479329351468 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 0] +
                 32.22887597610881 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 2, expression);
    expression = 1.8991047293938956 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.8991047293938956 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[0 * stride + 1] -
                 2.5321396391918607 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[0 * stride + 3] +
                 1.8991047293938956 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 1] +
                 1.8991047293938956 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[0 * stride + 1] -
                 2.5321396391918607 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[0 * stride + 3] -
                 11.394628376363375 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 1] -
                 11.394628376363375 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[2 * stride + 1] +
                 15.192837835151167 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 3, expression);
    expression = -0.30027482300412156 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.6005496460082431 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] +
                 2.4021985840329725 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.30027482300412156 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] +
                 2.4021985840329725 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.8007328613443242 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.30027482300412156 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.6005496460082431 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] +
                 2.4021985840329725 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.30027482300412156 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 2.4021985840329725 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.8007328613443242 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 4] +
                 1.8016489380247296 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] +
                 3.603297876049459 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] -
                 14.413191504197837 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2] +
                 1.8016489380247296 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0] -
                 14.413191504197837 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 2] +
                 4.804397168065946 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 4];
    atomicAdd(output + 2 * n_functions + 4, expression);
    expression = 1.8991047293938956 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.8991047293938956 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 2.5321396391918607 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 3] +
                 1.8991047293938956 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 1.8991047293938956 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] -
                 2.5321396391918607 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 3] -
                 11.394628376363375 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 11.394628376363375 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1] +
                 15.192837835151167 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 2 * n_functions + 5, expression);
    expression = 0.4476232774459556 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.685739664675734 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.4476232774459556 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] +
                 2.685739664675734 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.4476232774459556 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.685739664675734 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.4476232774459556 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 2.685739664675734 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 2] -
                 2.685739664675734 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] +
                 16.114437988054405 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 2] +
                 2.685739664675734 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0] -
                 16.114437988054405 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 2];
    atomicAdd(output + 2 * n_functions + 6, expression);
    expression = -1.674852942547621 * x_pairs[3 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 1] +
                 5.024558827642863 * x_pairs[3 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.674852942547621 * x_pairs[1 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 1] +
                 5.024558827642863 * x_pairs[1 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 1] +
                 10.049117655285729 * x_pairs[1 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 1] -
                 30.147352965857184 * x_pairs[1 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 2 * n_functions + 7, expression);
    expression = -0.592149936582833 * x_pairs[3 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[0 * stride + 0] +
                 3.552899619496998 * x_pairs[3 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.592149936582833 * x_pairs[3 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.592149936582833 * x_pairs[1 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[0 * stride + 0] +
                 3.552899619496998 * x_pairs[1 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.592149936582833 * x_pairs[1 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[0 * stride + 0] +
                 3.5528996194969986 * x_pairs[1 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[2 * stride + 0] -
                 21.31739771698199 * x_pairs[1 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[2 * stride + 0] +
                 3.5528996194969986 * x_pairs[1 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[2 * stride + 0];
    atomicAdd(output + 2 * n_functions + 8, expression);
    expression = -5.024558827642864 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 5.024558827642864 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] +
                 6.699411770190486 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0] -
                 6.699411770190486 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 0, expression);
    expression = -10.658698858490995 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] -
                 10.658698858490995 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 1] +
                 14.211598477987993 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 1] -
                 4.737199492662665 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 1, expression);
    expression = 1.8991047293938956 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.8991047293938956 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 0] -
                 11.394628376363375 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 2] +
                 1.8991047293938956 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.8991047293938956 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 0] -
                 11.394628376363375 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 2] -
                 2.5321396391918607 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 0] -
                 2.5321396391918607 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 0] +
                 15.192837835151167 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 2, expression);
    expression = 4.028609497013601 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[1 * stride + 3] +
                 4.028609497013601 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 3] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 1] * z_pairs[1 * stride + 3] -
                 5.371479329351469 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 1] -
                 5.371479329351469 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 3] * z_pairs[3 * stride + 1] +
                 7.161972439135291 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 1] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 3, expression);
    expression = -0.6369790906974141 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.2739581813948282 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.6369790906974141 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.6986109085264376 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 4] -
                 0.6369790906974141 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.2739581813948282 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.6369790906974141 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.6986109085264376 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 4] +
                 0.8493054542632188 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] +
                 1.6986109085264376 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0] -
                 6.79444363410575 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 2] +
                 0.8493054542632188 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[3 * stride + 0] -
                 6.79444363410575 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 2] +
                 2.264814544701917 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 4];
    atomicAdd(output + 3 * n_functions + 4, expression);
    expression = 4.028609497013601 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 3] +
                 4.028609497013601 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 3] -
                 5.371479329351469 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1] -
                 5.371479329351469 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 1] +
                 7.161972439135291 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 3 * n_functions + 5, expression);
    expression = 0.9495523646969478 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.6973141881816876 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.9495523646969478 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.6973141881816876 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.9495523646969478 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.6973141881816876 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.9495523646969478 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.6973141881816876 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.2660698195959303 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] +
                 7.596418917575583 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 2] +
                 1.2660698195959303 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[3 * stride + 0] -
                 7.596418917575583 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 2];
    atomicAdd(output + 3 * n_functions + 6, expression);
    expression = -3.552899619496998 * x_pairs[2 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[2 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 1] -
                 3.552899619496998 * x_pairs[0 * stride + 3] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[0 * stride + 1] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 1] +
                 4.737199492662665 * x_pairs[0 * stride + 3] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 1] -
                 14.211598477987993 * x_pairs[0 * stride + 1] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 3 * n_functions + 7, expression);
    expression = -1.256139706910716 * x_pairs[2 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[1 * stride + 0] +
                 7.536838241464296 * x_pairs[2 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[2 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[0 * stride + 4] *
                     y_pairs[3 * stride + 0] * z_pairs[1 * stride + 0] +
                 7.536838241464296 * x_pairs[0 * stride + 2] *
                     y_pairs[3 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[0 * stride + 0] *
                     y_pairs[3 * stride + 4] * z_pairs[1 * stride + 0] +
                 1.6748529425476215 * x_pairs[0 * stride + 4] *
                     y_pairs[1 * stride + 0] * z_pairs[3 * stride + 0] -
                 10.049117655285729 * x_pairs[0 * stride + 2] *
                     y_pairs[1 * stride + 2] * z_pairs[3 * stride + 0] +
                 1.6748529425476215 * x_pairs[0 * stride + 0] *
                     y_pairs[1 * stride + 4] * z_pairs[3 * stride + 0];
    atomicAdd(output + 3 * n_functions + 8, expression);
    expression = 0.7944525066428426 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.7944525066428426 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.5889050132856852 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 6.355620053142741 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 6.355620053142741 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 0.7944525066428426 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.7944525066428426 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] -
                 6.355620053142741 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 6.355620053142741 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] +
                 2.118540017714247 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0] -
                 2.118540017714247 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 0, expression);
    expression = 1.685288264333414 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.5617627547778047 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 3.370576528666828 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.1235255095556094 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 13.482306114667312 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 4.4941020382224375 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] +
                 1.685288264333414 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.5617627547778047 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1] -
                 13.482306114667312 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 4.4941020382224375 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 1] +
                 4.4941020382224375 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 1] -
                 1.4980340127408125 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 1, expression);
    expression = -0.30027482300412156 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.30027482300412156 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.8016489380247296 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.6005496460082431 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.6005496460082431 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.603297876049459 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] +
                 2.4021985840329725 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] +
                 2.4021985840329725 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] -
                 14.413191504197837 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.30027482300412156 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.30027482300412156 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.8016489380247296 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2] +
                 2.4021985840329725 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 2.4021985840329725 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] -
                 14.413191504197837 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.8007328613443242 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 0] -
                 0.8007328613443242 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 0] +
                 4.804397168065946 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 2];
    atomicAdd(output + 4 * n_functions + 2, expression);
    expression = -0.6369790906974141 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.6369790906974141 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 0.8493054542632188 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] -
                 1.2739581813948282 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.2739581813948282 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.6986109085264376 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3] +
                 5.095832725579313 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] +
                 5.095832725579313 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] -
                 6.79444363410575 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 3] -
                 0.6369790906974141 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.6369790906974141 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1] +
                 0.8493054542632188 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 3] +
                 5.095832725579313 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 5.095832725579313 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 1] -
                 6.79444363410575 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 3] -
                 1.6986109085264376 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 1] -
                 1.6986109085264376 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[4 * stride + 1] +
                 2.264814544701917 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[4 * stride + 3];
    atomicAdd(output + 4 * n_functions + 3, expression);
    expression = 0.10071523742534003 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20143047485068005 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8057218994027202 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.10071523742534003 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.8057218994027202 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.2685739664675734 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.20143047485068005 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.4028609497013601 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.6114437988054404 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.20143047485068005 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.6114437988054404 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.5371479329351468 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.8057218994027202 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.6114437988054404 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 6.445775195221762 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.8057218994027202 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] +
                 6.445775195221762 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] -
                 2.1485917317405874 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 4] +
                 0.10071523742534003 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.20143047485068005 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.8057218994027202 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.10071523742534003 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.8057218994027202 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.2685739664675734 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.8057218994027202 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.6114437988054404 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] +
                 6.445775195221762 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.8057218994027202 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0] +
                 6.445775195221762 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 2] -
                 2.1485917317405874 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 4] +
                 0.2685739664675734 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] +
                 0.5371479329351468 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0] -
                 2.1485917317405874 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 2] +
                 0.2685739664675734 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[4 * stride + 0] -
                 2.1485917317405874 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 2] +
                 0.7161972439135291 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 4];
    atomicAdd(output + 4 * n_functions + 4, expression);
    expression = -0.6369790906974141 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.6369790906974141 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.8493054542632188 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] -
                 1.2739581813948282 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.2739581813948282 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.6986109085264376 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] +
                 5.095832725579313 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 5.095832725579313 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] -
                 6.79444363410575 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3] -
                 0.6369790906974141 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.6369790906974141 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 0.8493054542632188 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3] +
                 5.095832725579313 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 5.095832725579313 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] -
                 6.79444363410575 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 3] -
                 1.6986109085264376 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1] -
                 1.6986109085264376 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 1] +
                 2.264814544701917 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 3];
    atomicAdd(output + 4 * n_functions + 5, expression);
    expression = -0.15013741150206078 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.9008244690123648 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.15013741150206078 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.9008244690123648 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.30027482300412156 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.8016489380247296 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.30027482300412156 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.8016489380247296 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.2010992920164862 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 7.206595752098918 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] -
                 1.2010992920164862 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] +
                 7.206595752098918 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] -
                 0.15013741150206078 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.9008244690123648 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.15013741150206078 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 0.9008244690123648 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.2010992920164862 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] -
                 7.206595752098918 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 1.2010992920164862 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0] +
                 7.206595752098918 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 2] -
                 0.4003664306721621 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] +
                 2.402198584032973 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 2] +
                 0.4003664306721621 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[4 * stride + 0] -
                 2.402198584032973 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 2];
    atomicAdd(output + 4 * n_functions + 6, expression);
    expression = 0.5617627547778047 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.685288264333414 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.1235255095556094 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.370576528666828 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 4.4941020382224375 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] +
                 13.482306114667312 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 0.5617627547778047 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.685288264333414 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] -
                 4.4941020382224375 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 13.482306114667312 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] +
                 1.4980340127408125 * x_pairs[0 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 1] -
                 4.4941020382224375 * x_pairs[0 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 1];
    atomicAdd(output + 4 * n_functions + 7, expression);
    expression = 0.19861312666071065 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.191678759964264 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.19861312666071065 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.3972262533214213 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.383357519928528 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3972262533214213 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 9.533430079714112 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.5889050132856852 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] +
                 0.19861312666071065 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.191678759964264 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.19861312666071065 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 9.533430079714112 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.5889050132856852 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0] +
                 0.5296350044285617 * x_pairs[0 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[4 * stride + 0] -
                 3.1778100265713705 * x_pairs[0 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[4 * stride + 0] +
                 0.5296350044285617 * x_pairs[0 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[4 * stride + 0];
    atomicAdd(output + 4 * n_functions + 8, expression);
    expression = -5.024558827642864 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 5.024558827642864 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] +
                 6.699411770190486 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 6.699411770190486 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 0, expression);
    expression = -10.658698858490995 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 10.658698858490995 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 3.552899619496998 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] +
                 14.211598477987993 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1] -
                 4.737199492662665 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 1, expression);
    expression = 1.8991047293938956 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.8991047293938956 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 11.394628376363375 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 1.8991047293938956 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 1.8991047293938956 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 11.394628376363375 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2] -
                 2.5321396391918607 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 0] -
                 2.5321396391918607 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 0] +
                 15.192837835151167 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 2];
    atomicAdd(output + 5 * n_functions + 2, expression);
    expression = 4.028609497013601 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3] +
                 4.028609497013601 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 3] -
                 5.371479329351469 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 1] -
                 5.371479329351469 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[3 * stride + 1] +
                 7.161972439135291 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[3 * stride + 3];
    atomicAdd(output + 5 * n_functions + 3, expression);
    expression = -0.6369790906974141 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.2739581813948282 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.6369790906974141 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.6986109085264376 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4] -
                 0.6369790906974141 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 1.2739581813948282 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.6369790906974141 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.095832725579313 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.6986109085264376 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 4] +
                 0.8493054542632188 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] +
                 1.6986109085264376 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] -
                 6.79444363410575 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2] +
                 0.8493054542632188 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0] -
                 6.79444363410575 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 2] +
                 2.264814544701917 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 4];
    atomicAdd(output + 5 * n_functions + 4, expression);
    expression = 4.028609497013601 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 4.028609497013601 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 4.028609497013601 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 5.371479329351469 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3] -
                 5.371479329351469 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 5.371479329351469 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1] +
                 7.161972439135291 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 3];
    atomicAdd(output + 5 * n_functions + 5, expression);
    expression = 0.9495523646969478 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.6973141881816876 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.9495523646969478 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.6973141881816876 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 0.9495523646969478 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 5.6973141881816876 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 0.9495523646969478 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 5.6973141881816876 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 1.2660698195959303 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] +
                 7.596418917575583 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 2] +
                 1.2660698195959303 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0] -
                 7.596418917575583 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 2];
    atomicAdd(output + 5 * n_functions + 6, expression);
    expression = -3.552899619496998 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 3.552899619496998 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] +
                 4.737199492662665 * x_pairs[1 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 1] -
                 14.211598477987993 * x_pairs[1 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 1];
    atomicAdd(output + 5 * n_functions + 7, expression);
    expression = -1.256139706910716 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 7.536838241464296 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 7.536838241464296 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 1.256139706910716 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 1.6748529425476215 * x_pairs[1 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[3 * stride + 0] -
                 10.049117655285729 * x_pairs[1 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[3 * stride + 0] +
                 1.6748529425476215 * x_pairs[1 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[3 * stride + 0];
    atomicAdd(output + 5 * n_functions + 8, expression);
    expression = -1.184299873165666 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 1.184299873165666 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 7.105799238993997 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 7.105799238993997 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 1.184299873165666 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.184299873165666 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] -
                 7.105799238993997 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 7.105799238993997 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 0, expression);
    expression = -2.5122794138214317 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.8374264712738105 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 15.073676482928592 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 5.024558827642864 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] +
                 2.5122794138214317 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.8374264712738105 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1] -
                 15.073676482928592 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 5.024558827642864 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 1, expression);
    expression = 0.4476232774459556 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] +
                 0.4476232774459556 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 2.685739664675734 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] -
                 2.685739664675734 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 0] -
                 2.685739664675734 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 0] +
                 16.114437988054405 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 2] -
                 0.4476232774459556 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.4476232774459556 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 2.685739664675734 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2] +
                 2.685739664675734 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 0] +
                 2.685739664675734 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 0] -
                 16.114437988054405 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 2];
    atomicAdd(output + 6 * n_functions + 2, expression);
    expression = 0.9495523646969478 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] +
                 0.9495523646969478 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 1.2660698195959303 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] -
                 5.6973141881816876 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 1] -
                 5.6973141881816876 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[2 * stride + 1] +
                 7.596418917575583 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[2 * stride + 3] -
                 0.9495523646969478 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 0.9495523646969478 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.2660698195959303 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 3] +
                 5.6973141881816876 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 1] +
                 5.6973141881816876 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[2 * stride + 1] -
                 7.596418917575583 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[2 * stride + 3];
    atomicAdd(output + 6 * n_functions + 3, expression);
    expression = -0.15013741150206078 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 0.30027482300412156 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 1.2010992920164862 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.15013741150206078 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.2010992920164862 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.4003664306721621 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.9008244690123648 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 1.8016489380247296 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] -
                 7.206595752098918 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 0.9008244690123648 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 7.206595752098918 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] +
                 2.402198584032973 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 4] +
                 0.15013741150206078 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.30027482300412156 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.2010992920164862 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.15013741150206078 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.2010992920164862 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.4003664306721621 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 4] -
                 0.9008244690123648 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] -
                 1.8016489380247296 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] +
                 7.206595752098918 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 0.9008244690123648 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0] +
                 7.206595752098918 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 2] -
                 2.402198584032973 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 4];
    atomicAdd(output + 6 * n_functions + 4, expression);
    expression = 0.9495523646969478 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 0.9495523646969478 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 1.2660698195959303 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] -
                 5.6973141881816876 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 5.6973141881816876 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 7.596418917575583 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 3] -
                 0.9495523646969478 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 0.9495523646969478 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.2660698195959303 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3] +
                 5.6973141881816876 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 5.6973141881816876 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1] -
                 7.596418917575583 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 3];
    atomicAdd(output + 6 * n_functions + 5, expression);
    expression = 0.2238116387229778 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.342869832337867 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] -
                 0.2238116387229778 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.342869832337867 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] -
                 1.342869832337867 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] +
                 8.057218994027203 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 2] +
                 1.342869832337867 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] -
                 8.057218994027203 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 2] -
                 0.2238116387229778 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.342869832337867 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.2238116387229778 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.342869832337867 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.342869832337867 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] -
                 8.057218994027203 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 2] -
                 1.342869832337867 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0] +
                 8.057218994027203 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 2];
    atomicAdd(output + 6 * n_functions + 6, expression);
    expression = -0.8374264712738105 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] +
                 2.5122794138214317 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 5.024558827642864 * x_pairs[2 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 1] -
                 15.073676482928592 * x_pairs[2 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 1] +
                 0.8374264712738105 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 2.5122794138214317 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] -
                 5.024558827642864 * x_pairs[0 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 1] +
                 15.073676482928592 * x_pairs[0 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 1];
    atomicAdd(output + 6 * n_functions + 7, expression);
    expression = -0.2960749682914165 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.776449809748499 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 0.2960749682914165 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] +
                 1.7764498097484993 * x_pairs[2 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[2 * stride + 0] -
                 10.658698858490995 * x_pairs[2 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[2 * stride + 0] +
                 1.7764498097484993 * x_pairs[2 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[2 * stride + 0] +
                 0.2960749682914165 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 1.776449809748499 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.2960749682914165 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.7764498097484993 * x_pairs[0 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[2 * stride + 0] +
                 10.658698858490995 * x_pairs[0 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[2 * stride + 0] -
                 1.7764498097484993 * x_pairs[0 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[2 * stride + 0];
    atomicAdd(output + 6 * n_functions + 8, expression);
    expression = 4.431244368585756 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 4.431244368585756 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] -
                 13.29373310575727 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 13.29373310575727 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 0, expression);
    expression = 9.400088826365067 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 3.133362942121689 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] -
                 28.200266479095202 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 9.400088826365067 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 1, expression);
    expression = -1.674852942547621 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 0] -
                 1.674852942547621 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 0] +
                 10.049117655285729 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 2] +
                 5.024558827642863 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 0] +
                 5.024558827642863 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 0] -
                 30.147352965857184 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 2];
    atomicAdd(output + 7 * n_functions + 2, expression);
    expression = -3.552899619496998 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 1] -
                 3.552899619496998 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[1 * stride + 1] +
                 4.737199492662665 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[1 * stride + 3] +
                 10.658698858490995 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[1 * stride + 1] -
                 14.211598477987993 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[1 * stride + 3];
    atomicAdd(output + 7 * n_functions + 3, expression);
    expression = 0.5617627547778047 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 1.1235255095556094 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] -
                 4.4941020382224375 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.5617627547778047 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 4.4941020382224375 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 1.4980340127408125 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 4] -
                 1.685288264333414 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 3.370576528666828 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] +
                 13.482306114667312 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 1.685288264333414 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 13.482306114667312 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2] -
                 4.4941020382224375 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 4];
    atomicAdd(output + 7 * n_functions + 4, expression);
    expression = -3.552899619496998 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 3.552899619496998 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] +
                 4.737199492662665 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 3] +
                 10.658698858490995 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 10.658698858490995 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1] -
                 14.211598477987993 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 3];
    atomicAdd(output + 7 * n_functions + 5, expression);
    expression = -0.8374264712738105 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] +
                 5.024558827642864 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 2] +
                 0.8374264712738105 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 5.024558827642864 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 2] +
                 2.5122794138214317 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] -
                 15.073676482928592 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 2] -
                 2.5122794138214317 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0] +
                 15.073676482928592 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 2];
    atomicAdd(output + 7 * n_functions + 6, expression);
    expression = 3.133362942121689 * x_pairs[3 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 1] -
                 9.400088826365067 * x_pairs[3 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 1] -
                 9.400088826365067 * x_pairs[1 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 1] +
                 28.200266479095202 * x_pairs[1 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 1];
    atomicAdd(output + 7 * n_functions + 7, expression);
    expression = 1.107811092146439 * x_pairs[3 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[1 * stride + 0] -
                 6.646866552878635 * x_pairs[3 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[1 * stride + 0] +
                 1.107811092146439 * x_pairs[3 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[1 * stride + 0] -
                 3.3234332764393173 * x_pairs[1 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[1 * stride + 0] +
                 19.940599658635904 * x_pairs[1 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[1 * stride + 0] -
                 3.3234332764393173 * x_pairs[1 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[1 * stride + 0];
    atomicAdd(output + 7 * n_functions + 8, expression);
    expression = 1.5666814710608448 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.5666814710608448 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] -
                 9.400088826365069 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 9.400088826365069 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] +
                 1.5666814710608448 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 1.5666814710608448 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 0, expression);
    expression = 3.3234332764393173 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.107811092146439 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] -
                 19.940599658635904 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 6.646866552878635 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] +
                 3.3234332764393173 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.107811092146439 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 1, expression);
    expression = -0.592149936582833 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.592149936582833 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.5528996194969986 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 2] +
                 3.552899619496998 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 0] +
                 3.552899619496998 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 0] -
                 21.31739771698199 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 2] -
                 0.592149936582833 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 0] -
                 0.592149936582833 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 0] +
                 3.5528996194969986 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 2];
    atomicAdd(output + 8 * n_functions + 2, expression);
    expression = -1.256139706910716 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.256139706910716 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.6748529425476215 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 1] * z_pairs[0 * stride + 3] +
                 7.536838241464296 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 1] +
                 7.536838241464296 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 3] * z_pairs[0 * stride + 1] -
                 10.049117655285729 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 1] * z_pairs[0 * stride + 3] -
                 1.256139706910716 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 1] -
                 1.256139706910716 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 3] * z_pairs[0 * stride + 1] +
                 1.6748529425476215 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 1] * z_pairs[0 * stride + 3];
    atomicAdd(output + 8 * n_functions + 3, expression);
    expression = 0.19861312666071065 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.3972262533214213 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.19861312666071065 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.5296350044285617 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 4] -
                 1.191678759964264 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.383357519928528 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] +
                 9.533430079714112 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 1.191678759964264 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 9.533430079714112 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 3.1778100265713705 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 4] +
                 0.19861312666071065 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 0.3972262533214213 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.19861312666071065 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.5889050132856852 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2] +
                 0.5296350044285617 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 4];
    atomicAdd(output + 8 * n_functions + 4, expression);
    expression = -1.256139706910716 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.256139706910716 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.6748529425476215 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 3] +
                 7.536838241464296 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 7.536838241464296 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] -
                 10.049117655285729 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 3] -
                 1.256139706910716 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 1.256139706910716 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.6748529425476215 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 3];
    atomicAdd(output + 8 * n_functions + 5, expression);
    expression = -0.2960749682914165 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7764498097484993 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.2960749682914165 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.7764498097484993 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 2] +
                 1.776449809748499 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] -
                 10.658698858490995 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 2] -
                 1.776449809748499 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 10.658698858490995 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 2] -
                 0.2960749682914165 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] +
                 1.7764498097484993 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 2] +
                 0.2960749682914165 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0] -
                 1.7764498097484993 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 2];
    atomicAdd(output + 8 * n_functions + 6, expression);
    expression = 1.107811092146439 * x_pairs[4 * stride + 3] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.3234332764393173 * x_pairs[4 * stride + 1] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 1] -
                 6.646866552878635 * x_pairs[2 * stride + 3] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 1] +
                 19.940599658635904 * x_pairs[2 * stride + 1] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 1] +
                 1.107811092146439 * x_pairs[0 * stride + 3] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 1] -
                 3.3234332764393173 * x_pairs[0 * stride + 1] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 1];
    atomicAdd(output + 8 * n_functions + 7, expression);
    expression = 0.3916703677652112 * x_pairs[4 * stride + 4] *
                     y_pairs[0 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.3500222065912673 * x_pairs[4 * stride + 2] *
                     y_pairs[0 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3916703677652112 * x_pairs[4 * stride + 0] *
                     y_pairs[0 * stride + 4] * z_pairs[0 * stride + 0] -
                 2.3500222065912673 * x_pairs[2 * stride + 4] *
                     y_pairs[2 * stride + 0] * z_pairs[0 * stride + 0] +
                 14.100133239547603 * x_pairs[2 * stride + 2] *
                     y_pairs[2 * stride + 2] * z_pairs[0 * stride + 0] -
                 2.3500222065912673 * x_pairs[2 * stride + 0] *
                     y_pairs[2 * stride + 4] * z_pairs[0 * stride + 0] +
                 0.3916703677652112 * x_pairs[0 * stride + 4] *
                     y_pairs[4 * stride + 0] * z_pairs[0 * stride + 0] -
                 2.3500222065912673 * x_pairs[0 * stride + 2] *
                     y_pairs[4 * stride + 2] * z_pairs[0 * stride + 0] +
                 0.3916703677652112 * x_pairs[0 * stride + 0] *
                     y_pairs[4 * stride + 4] * z_pairs[0 * stride + 0];
    atomicAdd(output + 8 * n_functions + 8, expression);
  }
}
} // namespace ovlp
