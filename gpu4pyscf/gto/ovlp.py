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

from gpu4pyscf.lib.cupy_helper import load_library
from pyscf.gto import NPRIM_OF, NCTR_OF, ANG_OF, PTR_EXP, PTR_COEFF, ATOM_OF, PTR_COORD
from pyscf.gto.mole import cart2sph


libovlp = load_library('libovlp')


def cast_to_pointer(array):
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError('Invalid array type')


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


def get_ovlp_for_single_mol(mol):
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

    n_primitives = decontracted_basis.shape[0]
    sort_index_by_angular = np.argsort(decontracted_basis[:, ANG_OF])
    decontracted_basis = decontracted_basis[sort_index_by_angular]
    shell_to_ao = cp.asarray(shell_to_ao[sort_index_by_angular], dtype=cp.int32)

    left_shells, right_shells = np.triu_indices(n_primitives)
    n_pairs = len(left_shells)
    angular_pairs = np.zeros((2, n_pairs), dtype=np.int32)
    angular_pairs[0] = decontracted_basis[left_shells, ANG_OF]
    angular_pairs[1] = decontracted_basis[right_shells, ANG_OF]

    groups, indices = unique_with_multiple_keys(angular_pairs.T)
    sorted_pairs = []
    for i, group in enumerate(groups):
        pairs = np.where(indices == i)[0]
        left_shells_in_this_group = cp.asarray(left_shells[pairs], dtype=cp.int32)
        right_shells_in_this_group = cp.asarray(right_shells[pairs], dtype=cp.int32)
        pairs = left_shells_in_this_group * n_primitives + right_shells_in_this_group
        sorted_pairs.append({'angular_pairs': group, 'primitive_pairs': pairs})

    atm = cp.asarray(mol._atm, dtype=cp.int32)
    env = cp.asarray(mol._env, dtype=cp.double)

    result = cp.zeros((n_functions, n_functions))
    decontracted_basis = cp.asarray(decontracted_basis, dtype=cp.int32)
    for pairs in sorted_pairs:
        i_angular, j_angular = pairs['angular_pairs']
        libovlp.overlap(
            cast_to_pointer(result),
            cast_to_pointer(pairs['primitive_pairs']),
            ctypes.c_int(len(pairs['primitive_pairs'])),
            ctypes.c_int(n_primitives),
            cast_to_pointer(shell_to_ao),
            ctypes.c_int(n_functions),
            cast_to_pointer(atm),
            ctypes.c_int(atm.size),
            cast_to_pointer(decontracted_basis),
            ctypes.c_int(decontracted_basis.size),
            cast_to_pointer(env),
            ctypes.c_int(env.size),
            ctypes.c_int(1),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
        )

    return result + result.T


def create_ovlp_plan(atms, bases, envs):
    assert len(atms.shape) == len(bases.shape)
    assert len(envs.shape) == 2
    assert atms.shape[0] == bases.shape[0] == envs.shape[0]
    assert np.all(bases[:, :, ANG_OF] == bases[0, :, ANG_OF])
    assert np.all(bases[:, :, NCTR_OF] == bases[0, :, NCTR_OF])
    assert np.all(bases[:, :, NPRIM_OF] == bases[0, :, NPRIM_OF])

    n_configurations = atms.shape[0]
    n_contracted = bases[0, :, NCTR_OF]
    n_primitives_per_shell = bases[0, :, NPRIM_OF]
    decontracted_basis = np.repeat(bases, n_contracted, axis=-2)
    decontracted_basis[:, :, NCTR_OF] = 1
    coeff_offset = np.concatenate([np.arange(i) * n for i, n in zip(n_contracted, n_primitives_per_shell)])
    decontracted_basis[:, :, PTR_COEFF] += coeff_offset
    shell_to_ao = make_loc(decontracted_basis[0], 'sph')
    n_functions = shell_to_ao[-1]
    shell_to_ao = shell_to_ao[:-1]

    n_primitives_per_shell = np.repeat(n_primitives_per_shell, n_contracted)
    decontracted_basis = np.repeat(decontracted_basis, n_primitives_per_shell, axis=-2)
    primitive_offset = np.concatenate([np.arange(i) for i in n_primitives_per_shell])
    decontracted_basis[:, :, NPRIM_OF] = 1
    decontracted_basis[:, :, PTR_COEFF] += primitive_offset
    decontracted_basis[:, :, PTR_EXP] += primitive_offset
    shell_to_ao = np.repeat(shell_to_ao, n_primitives_per_shell)

    angulars = decontracted_basis[0, :, ANG_OF]

    n_primitives = decontracted_basis.shape[-2]
    sort_index_by_angular = np.argsort(angulars)
    angulars = angulars[sort_index_by_angular]
    spikes = angulars[1:] - angulars[:-1]
    changed_indices = np.where(spikes)[0] + 1
    max_angular = len(changed_indices)

    grouped_primitives_ranges = np.zeros((max_angular + 1, 2), dtype=np.int32)
    grouped_primitives_ranges[1:, 0] = changed_indices
    grouped_primitives_ranges[:-1, 1] = changed_indices
    grouped_primitives_ranges[-1, 1] = n_primitives
    decontracted_basis = decontracted_basis[:, sort_index_by_angular]

    atms = cp.asarray(atms, dtype=cp.int32)
    bases = cp.asarray(decontracted_basis, dtype=cp.int32)
    envs = cp.asarray(envs, dtype=cp.double)
    shell_to_ao = cp.asarray(shell_to_ao[sort_index_by_angular], dtype=cp.int32)

    pairs = []
    for i_angular in range(max_angular + 1):
        i_range = grouped_primitives_ranges[i_angular]
        for j_angular in range(i_angular, max_angular + 1):
            j_range = grouped_primitives_ranges[j_angular]

            if i_angular == j_angular:
                left_pairs, right_pairs = cp.triu_indices(i_range[1] - i_range[0])
                left_pairs += i_range[0]
                right_pairs += j_range[0]
                pair_indices = cp.asarray(left_pairs * n_primitives + right_pairs, dtype=cp.int32).flatten()
            else:
                left_pairs = cp.arange(*i_range, dtype=cp.int32)
                right_pairs = cp.arange(*j_range, dtype=cp.int32)
                pair_indices = cp.asarray(left_pairs[:, None] * n_primitives + right_pairs[None, :], dtype=cp.int32)

            pairs.append((i_angular, j_angular, pair_indices))

    plan = {
        'atms': atms,
        'bases': bases,
        'envs': envs,
        'shell_to_ao': shell_to_ao,
        'n_configurations': n_configurations,
        'n_functions': n_functions,
        'n_primitives': n_primitives,
        'pairs': pairs,
    }

    return plan


def get_ovlp(plan):
    result = cp.zeros((plan['n_configurations'], plan['n_functions'], plan['n_functions']))

    for i_angular, j_angular, pair_indices in plan['pairs']:
        libovlp.overlap(
            cast_to_pointer(result),
            cast_to_pointer(pair_indices),
            ctypes.c_int(pair_indices.size),
            ctypes.c_int(plan['n_primitives']),
            cast_to_pointer(plan['shell_to_ao']),
            ctypes.c_int(plan['n_functions']),
            cast_to_pointer(plan['atms']),
            ctypes.c_int(plan['atms'][0].size),
            cast_to_pointer(plan['bases']),
            ctypes.c_int(plan['bases'][0].size),
            cast_to_pointer(plan['envs']),
            ctypes.c_int(plan['envs'][0].size),
            ctypes.c_int(plan['n_configurations']),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
        )

    return result + result.transpose(0, 2, 1)


def get_ovlp_gradient(plan):
    result = cp.zeros((plan['n_configurations'], 3, plan['n_functions'], plan['n_functions']))

    for i_angular, j_angular, pair_indices in plan['pairs']:
        libovlp.overlap_gradient(
            cast_to_pointer(result),
            cast_to_pointer(pair_indices),
            ctypes.c_int(pair_indices.size),
            ctypes.c_int(plan['n_primitives']),
            cast_to_pointer(plan['shell_to_ao']),
            ctypes.c_int(plan['n_functions']),
            cast_to_pointer(plan['atms']),
            ctypes.c_int(plan['atms'][0].size),
            cast_to_pointer(plan['bases']),
            ctypes.c_int(plan['bases'][0].size),
            cast_to_pointer(plan['envs']),
            ctypes.c_int(plan['envs'][0].size),
            ctypes.c_int(plan['n_configurations']),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
        )

    return result.transpose(0, 1, 3, 2) - result


def get_dipole(plan, reference_point=(0, 0, 0)):
    result = cp.zeros((plan['n_configurations'], 3, plan['n_functions'], plan['n_functions']))

    for i_angular, j_angular, pair_indices in plan['pairs']:
        libovlp.dipole(
            cast_to_pointer(result),
            cast_to_pointer(pair_indices),
            ctypes.c_int(pair_indices.size),
            ctypes.c_int(plan['n_primitives']),
            cast_to_pointer(plan['shell_to_ao']),
            ctypes.c_int(plan['n_functions']),
            cast_to_pointer(plan['atms']),
            ctypes.c_int(plan['atms'][0].size),
            cast_to_pointer(plan['bases']),
            ctypes.c_int(plan['bases'][0].size),
            cast_to_pointer(plan['envs']),
            ctypes.c_int(plan['envs'][0].size),
            ctypes.c_int(plan['n_configurations']),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            ctypes.c_double(reference_point[0]),
            ctypes.c_double(reference_point[1]),
            ctypes.c_double(reference_point[2]),
        )

    return result + result.transpose(0, 1, 3, 2)


def get_quadrupole(plan, reference_point=(0, 0, 0)):
    result = cp.zeros((plan['n_configurations'], 9, plan['n_functions'], plan['n_functions']))

    for i_angular, j_angular, pair_indices in plan['pairs']:
        assert i_angular <= j_angular
        libovlp.quadrupole(
            cast_to_pointer(result),
            cast_to_pointer(pair_indices),
            ctypes.c_int(pair_indices.size),
            ctypes.c_int(plan['n_primitives']),
            cast_to_pointer(plan['shell_to_ao']),
            ctypes.c_int(plan['n_functions']),
            cast_to_pointer(plan['atms']),
            ctypes.c_int(plan['atms'][0].size),
            cast_to_pointer(plan['bases']),
            ctypes.c_int(plan['bases'][0].size),
            cast_to_pointer(plan['envs']),
            ctypes.c_int(plan['envs'][0].size),
            ctypes.c_int(plan['n_configurations']),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            ctypes.c_double(reference_point[0]),
            ctypes.c_double(reference_point[1]),
            ctypes.c_double(reference_point[2]),
        )
    result += result.transpose(0, 1, 3, 2)

    result[:, [3, 6, 7]] = result[:, [1, 2, 5]]

    return result
