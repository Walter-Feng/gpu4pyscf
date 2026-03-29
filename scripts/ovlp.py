import numpy as np
import math
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


def gradient(cartesian_functions, gradient_operator):
    if np.all(gradient_operator == 0):
        return cartesian_functions
    else:
        if np.any(gradient_operator < 0):
            return []
        else:
            result = []
            for i in cartesian_functions:
                pass


def position_operator(components: list, order: int, axis: str) -> list:
    if order == 0:
        return components

    result = []
    for component in components:
        for i in range(order + 1):
            binomial_coeff = math.comb(order, i)
            if binomial_coeff == 1:
                binomial_coeff = []
            else:
                binomial_coeff = [binomial_coeff]
            offset_term = [axis + '_offset'] * i
            result.append((binomial_coeff + offset_term + component[0], component[1], component[2] + order - i))

    return result


def position_operator_list(order: int):
    for z in range(order + 1):
        for y in range(order - z + 1):
            x = order - z - y
            yield (x, y, z)


def gradient_operator(components: list, orders: tuple, axis: str) -> list:
    if orders == (0, 0):
        return components
    if orders[0] < 0 or orders[1] < 0:
        return []

    bra_order = orders[0]
    ket_order = orders[1]

    result = []
    for component in components:
        coeff = component[0]
        bra_power = component[1]
        ket_power = component[2]

        if bra_order > 0:
            new_order = (orders[0] - 1, orders[1])
            if bra_power > 0:
                term_a = ([bra_power] + coeff, bra_power - 1, ket_power)
                result += gradient_operator([term_a], new_order, axis)

            term_b = ([2] + coeff + ['i_exponent'], bra_power + 1, ket_power)
            result += gradient_operator([term_b], new_order, axis)

        if ket_order > 0 and bra_order == 0:
            new_order = (orders[0], orders[1] - 1)
            if ket_power > 0:
                term_a = ([ket_power] + coeff, bra_power, ket_power - 1)
                result += gradient_operator([term_a], new_order, axis)

            term_b = ([2] + coeff + ['j_exponent'], bra_power, ket_power + 1)
            result += gradient_operator([term_b], new_order, axis)

    return result


def gradient_operator_list(orders: tuple):
    for x_bra, y_bra, z_bra in position_operator_list(orders[0]):
        for x_ket, y_ket, z_ket in position_operator_list(orders[1]):
            yield (x_bra, x_ket), (y_bra, y_ket), (z_bra, z_ket)


def merge_coefficients(list_of_expressions: list):
    merged = ''
    numerical = 1
    for expr in list_of_expressions:
        if isinstance(expr, int) or isinstance(expr, float):
            numerical *= expr
        else:
            assert isinstance(expr, str)
            merged += expr + '*'

    if numerical == 1:
        return merged
    else:
        return str(numerical) + '*' + merged


def sum_of_components_to_string(list_of_expressions, axis):
    array_name = 'X_pairs[{}]'.replace('X', axis)

    converted_exprs = []
    for expr in list_of_expressions:
        coeff = merge_coefficients(expr[0])
        flattened_index = '{0}*stride+{1}'.format(expr[1], expr[2])
        converted_exprs.append(coeff + array_name.format(flattened_index))

    assert len(converted_exprs) > 0

    if len(converted_exprs) > 1:
        return '({})'.format(' + '.join(converted_exprs))
    else:
        return converted_exprs[0]


def spherical_string(i_angular, j_angular, position_order=0):
    i_cartesian = cartesian(i_angular)
    spherical_coeff_i = cart2sph(i_angular, normalized='sp').T
    j_cartesian = cartesian(j_angular)
    spherical_coeff_j = cart2sph(j_angular, normalized='sp').T

    snippet = ''
    stride = position_order + j_angular + 1

    n_position_operators = (position_order + 1) * (position_order + 2) // 2

    for i_operator, r_operator in enumerate(position_operator_list(position_order)):
        for i, i_spherical_coeff in enumerate(spherical_coeff_i):
            for j, j_spherical_coeff in enumerate(spherical_coeff_j):
                expr = ''
                for i_cartesian_coeff, i_function in zip(i_spherical_coeff, i_cartesian):
                    for j_cartesian_coeff, j_function in zip(j_spherical_coeff, j_cartesian):
                        coeff = i_cartesian_coeff * j_cartesian_coeff
                        if abs(coeff) > 1e-15:
                            term = str(coeff)
                            for xyz_index, (xyz_string, r) in enumerate(zip(['x', 'y', 'z'], r_operator)):
                                component = sum_of_components_to_string(
                                    position_operator(
                                        [([], i_function[xyz_index], j_function[xyz_index])], r, xyz_string
                                    ),
                                    xyz_string,
                                )

                                term += '*' + component

                            if coeff > 0:
                                term = '+' + term

                            expr += term

                expr = 'expression = {2}; atomicAdd(output + {0} * n_functions + {1}, expression);'.format(i, j, expr)
                expr = expr.replace('1.0*', '').replace('= +', '= ')
                snippet += expr
        if i_operator != n_position_operators - 1:
            snippet += '\noutput += n_functions * n_functions;\n'

    return 'if constexpr(i_angular == {} && j_angular == {}){{{}}}'.format(i_angular, j_angular, snippet)


for i in range(5):
    for j in range(5):
        print(spherical_string(i, j, 0))
