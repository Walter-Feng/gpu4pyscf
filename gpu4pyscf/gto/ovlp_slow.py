#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes
import numpy as np
import cupy as cp

from pyscf.gto.moleintor import make_loc

from pyscf.gto import NPRIM_OF, NCTR_OF, ANG_OF, PTR_EXP, PTR_COEFF, ATOM_OF, PTR_COORD
from pyscf.gto.mole import cart2sph


def cast_to_pointer(array):
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError('Invalid array type')


def cartesian(angular):
    result = []
    for i in range(angular, -1, -1):
        for j in range(angular - i, -1, -1):
            result.append((i, j, angular - i - j))

    return np.array(result)


def VRR(result, total_angular, a00, pair_exponent, another_exponent, shift_to_here):
    result[0] = a00
    if total_angular > 0:
        result[1] = -another_exponent / pair_exponent * shift_to_here * a00

    for i in range(1, total_angular):
        result[i + 1] = (
            0.5 * i / pair_exponent * result[i - 1] - another_exponent / pair_exponent * shift_to_here * result[i]
        )


def HRR(result, i_angular, j_angular, shift_to_here):
    for i in range(i_angular):
        for j in range(i_angular + j_angular - i - 1, -1, -1):
            result[(i + 1) * (j_angular + 1) + j] = (
                result[i * (j_angular + 1) + j + 1] + shift_to_here * result[i * (j_angular + 1) + j]
            )


def RR(i_angular, j_angular, a00, pair_exponent, i_exponent, i_to_j):
    result = np.zeros((i_angular + 1) * (j_angular + 1))
    VRR(result, i_angular + j_angular, a00, pair_exponent, i_exponent, i_to_j)
    HRR(result, i_angular, j_angular, i_to_j)

    return result.reshape((i_angular + 1, j_angular + 1))


def common_fac_sp(angular):
    if angular == 0:
        return 0.282094791773878143
    if angular == 1:
        return 0.488602511902919921

    return 1.0


def unique_with_multiple_keys(x):
    # This function expands the previous function to handle multiple keys
    # shaped as [ (1, 2), (3, -4), ....]
    assert type(x) is np.ndarray and (x.dtype == np.int32 or x.dtype == np.int64) and x.ndim == 2

    x = x.T
    n = x.shape[-1]

    inverse_sort = np.zeros(n, dtype=np.int64)
    if n <= 1:
        return x.T, inverse_sort

    sort_index = np.lexsort(x)
    inverse_sort[sort_index] = np.arange(0, n, dtype=np.int64)
    x = x[:, sort_index].T

    mask = np.empty(n, dtype=np.bool_)
    mask[0] = True
    mask[1:] = np.any(x[1:] != x[:-1], axis=-1)

    x = x[mask]
    inverse_unique = np.cumsum(mask, dtype=np.int64) - 1

    return x, inverse_unique[inverse_sort]


def get_ovlp(mol):
    n_contracted = mol._bas[:, NCTR_OF]
    n_primitives_per_shell = mol._bas[:, NPRIM_OF]
    decontracted_basis = np.repeat(mol._bas, n_contracted, axis=0)
    decontracted_basis[:, NCTR_OF] = 1
    coeff_offset = np.concatenate([np.arange(i) * n for i, n in zip(n_contracted, n_primitives_per_shell)])
    decontracted_basis[:, PTR_COEFF] += coeff_offset
    shell_to_ao = make_loc(decontracted_basis, 'sph')
    n_functions = shell_to_ao[-1]
    shell_to_ao = shell_to_ao[:-1]

    n_primitives_per_shell = np.repeat(n_primitives_per_shell, n_contracted)
    decontracted_basis = np.repeat(decontracted_basis, n_primitives_per_shell, axis=0)
    primitive_offset = np.concatenate([np.arange(i) for i in n_primitives_per_shell])
    decontracted_basis[:, NPRIM_OF] = 1
    decontracted_basis[:, PTR_COEFF] += primitive_offset
    decontracted_basis[:, PTR_EXP] += primitive_offset
    shell_to_ao = np.repeat(shell_to_ao, n_primitives_per_shell)

    sort_index_by_angular = np.argsort(decontracted_basis[:, ANG_OF])
    decontracted_basis = decontracted_basis[sort_index_by_angular]
    shell_to_ao = shell_to_ao[sort_index_by_angular]

    n_primitives = decontracted_basis.shape[0]
    left_shells, right_shells = np.triu_indices(n_primitives)
    n_pairs = len(left_shells)
    angular_pairs = np.zeros((2, n_pairs), dtype=np.int32)
    angular_pairs[0] = decontracted_basis[left_shells, ANG_OF]
    angular_pairs[1] = decontracted_basis[right_shells, ANG_OF]

    groups, indices = unique_with_multiple_keys(angular_pairs.T)
    sorted_pairs = []
    for i, group in enumerate(groups):
        pairs = np.where(indices == i)[0]
        left_shells_in_this_group = np.asarray(left_shells[pairs], dtype=np.int32)
        right_shells_in_this_group = np.asarray(right_shells[pairs], dtype=np.int32)
        sorted_pairs.append(
            {
                'angular_pairs': group,
                'left_shells': left_shells_in_this_group,
                'right_shells': right_shells_in_this_group,
            }
        )

    atm = mol._atm
    env = mol._env

    result = np.zeros((n_functions, n_functions))
    for pairs in sorted_pairs:
        i_angular, j_angular = pairs['angular_pairs']

        for l_shell, r_shell in zip(pairs['left_shells'], pairs['right_shells']):
            alpha = env[decontracted_basis[l_shell, PTR_EXP]]
            beta = env[decontracted_basis[r_shell, PTR_EXP]]

            c1 = env[decontracted_basis[l_shell, PTR_COEFF]] * common_fac_sp(i_angular)
            c2 = env[decontracted_basis[r_shell, PTR_COEFF]] * common_fac_sp(j_angular)

            l_atom = decontracted_basis[l_shell, ATOM_OF]
            r_atom = decontracted_basis[r_shell, ATOM_OF]

            i_coord_offset = atm[l_atom, PTR_COORD]
            j_coord_offset = atm[r_atom, PTR_COORD]

            i_center = env[i_coord_offset : i_coord_offset + 3]
            j_center = env[j_coord_offset : j_coord_offset + 3]

            pair_center = alpha * i_center + beta * j_center
            pair_exponent = alpha + beta

            prefactor = np.exp(-alpha * beta / pair_exponent * np.sum((i_center - j_center) ** 2))

            i_function_index = shell_to_ao[l_shell]
            j_function_index = shell_to_ao[r_shell]

            i_to_j = j_center - i_center

            if l_shell == r_shell:
                prefactor *= 0.5

            if i_angular == 0 and j_angular == 0:
                a00 = prefactor * (np.pi / pair_exponent) ** 1.5
                ovlp = c1 * c2 * a00

                result[i_function_index, j_function_index] += ovlp
                result[j_function_index, i_function_index] += ovlp

            else:
                i_cartesian = cartesian(i_angular)
                spherical_coeff_i = cart2sph(i_angular, normalized='sp').T
                j_cartesian = cartesian(j_angular)
                spherical_coeff_j = cart2sph(j_angular, normalized='sp').T

                RRx = RR(
                    i_angular,
                    j_angular,
                    prefactor * c1 * c2 * (np.pi / pair_exponent) ** 1.5,
                    pair_exponent,
                    alpha,
                    i_to_j[0],
                )
                RRy = RR(i_angular, j_angular, 1, pair_exponent, alpha, i_to_j[1])
                RRz = RR(i_angular, j_angular, 1, pair_exponent, alpha, i_to_j[2])

                for i, i_spherical_coeff in enumerate(spherical_coeff_i):
                    for j, j_spherical_coeff in enumerate(spherical_coeff_j):
                        ovlp = 0
                        for i_cartesian_coeff, i_function in zip(i_spherical_coeff, i_cartesian):
                            for j_cartesian_coeff, j_function in zip(j_spherical_coeff, j_cartesian):
                                x_component = RRx[i_function[0], j_function[0]]
                                y_component = RRy[i_function[1], j_function[1]]
                                z_component = RRz[i_function[2], j_function[2]]

                                ovlp += i_cartesian_coeff * j_cartesian_coeff * x_component * y_component * z_component

                        result[i_function_index + i, j_function_index + j] += ovlp
                        result[j_function_index + j, i_function_index + i] += ovlp

    return result
