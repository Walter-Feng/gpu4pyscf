import numpy as np
from pyscf.gto.mole import cart2sph


def cartesian(angular):
    result = []
    for i in range(angular, -1, -1):
        for j in range(angular - i, -1, -1):
            result.append((i, j, angular - i - j))

    return np.array(result)


def HRR_string(i_angular, j_angular):
    body = ''
    for i in range(i_angular):
        for j in range(i_angular + j_angular - i - 1, -1, -1):
            first_index = (i + 1) * (j_angular + 1) + j
            second_index = i * (j_angular + 1) + j + 1
            third_index = i * (j_angular + 1) + j
            body += 'result[{}] = result[{}] + shift_to_here * result[{}];'.format(
                first_index, second_index, third_index
            )

    if body != '':
        return 'if constexpr(i_angular == {} && j_angular == {}) {{ {} }}'.format(i_angular, j_angular, body)
    else:
        return ''


def spherical_string(i_angular, j_angular):
    i_cartesian = cartesian(i_angular)
    spherical_coeff_i = cart2sph(i_angular, normalized='sp').T
    j_cartesian = cartesian(j_angular)
    spherical_coeff_j = cart2sph(j_angular, normalized='sp').T

    snippet = ''
    for i, i_spherical_coeff in enumerate(spherical_coeff_i):
        for j, j_spherical_coeff in enumerate(spherical_coeff_j):
            expr = ''
            for i_cartesian_coeff, i_function in zip(i_spherical_coeff, i_cartesian):
                for j_cartesian_coeff, j_function in zip(j_spherical_coeff, j_cartesian):
                    coeff = i_cartesian_coeff * j_cartesian_coeff
                    if abs(coeff) > 1e-15:
                        term = '{} * x_pairs[{}] * y_pairs[{}] * z_pairs[{}]'.format(
                            coeff,
                            i_function[0] * (j_angular + 1) + j_function[0],
                            i_function[1] * (j_angular + 1) + j_function[1],
                            i_function[2] * (j_angular + 1) + j_function[2],
                        )

                        if not '-' in term:
                            term = ' + ' + term

                        expr += term

            expr = 'expression = {2}; atomicAdd(output + {0} * n_functions + {1}, expression); atomicAdd(output_transpose + {1} * n_functions + {0}, expression); '.format(
                i, j, expr
            )
            expr = expr.replace('1.0 *', '').replace('= +', '= ')
            snippet += expr

    return 'if constexpr(i_angular == {} && j_angular == {}){{{}}}'.format(i_angular, j_angular, snippet)
