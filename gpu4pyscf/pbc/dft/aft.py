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

import pyscf.pbc.gto as gto
from pyscf.pbc.gto.eval_gto import get_lattice_Ls
from pyscf.gto.moleintor import make_loc
from pyscf.pbc.tools import super_cell
from pyscf import lib

from gpu4pyscf.dft import numint as mol_numint
from gpu4pyscf.pbc.dft import RKS
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc.tools import pbc as pbc_tools
import gpu4pyscf.pbc.dft.multigrid_v2 as multigrid_v2
import gpu4pyscf.pbc.dft.gen_grid as gen_grid
from gpu4pyscf.lib.cupy_helper import tag_array, load_library
from pyscf.gto import NPRIM_OF, NCTR_OF, ANG_OF, PTR_EXP, PTR_COEFF, ATOM_OF, PTR_COORD
from pyscf.gto.mole import cart2sph


libaft = load_library('libaft')


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


def cartesian(angular):
    result = []
    for i in range(angular, -1, -1):
        for j in range(angular - i, -1, -1):
            result.append((i, j, angular - i - j))

    return np.array(result)


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


def get_ovlp_ref(mol):
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
    shell_to_ao = cp.asarray(shell_to_ao[sort_index_by_angular], dtype=cp.int32)

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
        left_shells_in_this_group = cp.asarray(left_shells[pairs], dtype=cp.int32)
        right_shells_in_this_group = cp.asarray(right_shells[pairs], dtype=cp.int32)
        pairs = left_shells_in_this_group * n_primitives + right_shells_in_this_group
        sorted_pairs.append({'angular_pairs': group, 'primitive_pairs': pairs})

    atm = cp.asarray(mol._atm, dtype=cp.int32)
    env = cp.asarray(mol._env, dtype=cp.double)

    result = cp.zeros((n_functions, n_functions))
    for pairs in sorted_pairs:
        i_angular, j_angular = pairs['angular_pairs']
        libaft.overlap(
            multigrid_v2.cast_to_pointer(result),
            multigrid_v2.cast_to_pointer(pairs['primitive_pairs']),
            ctypes.c_int(len(pairs['primitive_pairs'])),
            ctypes.c_int(n_primitives),
            multigrid_v2.cast_to_pointer(shell_to_ao),
            ctypes.c_int(n_functions),
            multigrid_v2.cast_to_pointer(atm),
            ctypes.c_int(atm.size),
            multigrid_v2.cast_to_pointer(decontracted_basis),
            ctypes.c_int(decontracted_basis.size),
            multigrid_v2.cast_to_pointer(env),
            ctypes.c_int(env.size),
            ctypes.c_int(1),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
        )

    return result


from pyscf.gto import Mole

mol = Mole(
    atom="""O     0.      0.      0.    
            He     1.      0.      0.    
            He     1.      2.      0.    
         """,
    basis='cc-pvqz',
    verbose=5,
)
mol.build()
get_ovlp(mol)
assert 0


class AFTDFNumInt(pbc_numint.NumInt):
    def __init__(self, cell: gto.Cell):
        libaft.update_reciprocal_lattice_vectors(
            multigrid_v2.cast_to_pointer(cp.asarray(cell.reciprocal_vectors().T, order='C')),
        )
        self.cell = cell
        if self.cell.precision:
            self.precision = self.cell.precision
        else:
            self.precision = 1e-8

        self.precision *= 1e-2
        self.mesh = cell.mesh
        self.grid = gen_grid.UniformGrids(cell)
        self.tasks = None
        self.sorted_gaussian_pairs = None
        Gv = pbc_tools._get_Gv(cell, cell.mesh)
        self.coulG = pbc_tools.get_coulG(cell, Gv=Gv)
        self.build()

    def build(self):
        contracted_cell = self.cell
        cell, transform_coeff = contracted_cell.decontract_basis(to_cart=True)
        sorted_transform_coeff, transform_shape = multigrid_v2.sort_contraction_coefficients(transform_coeff)

        neighboring_images = get_lattice_Ls(contracted_cell)
        images_sort_index = np.argsort(np.linalg.norm(neighboring_images, axis=1))
        neighboring_images = neighboring_images[images_sort_index]

        self.primitive = cell
        self.transform_coeff = sorted_transform_coeff
        self.shell_to_ao = cp.asarray(make_loc(cell._bas, 'cart'), dtype=cp.int32)
        self.n_primitives = len(cell._bas)
        self.n_primitive_functions = transform_shape[0]
        self.n_functions = transform_shape[1]

        self.atm = cp.asarray(cell._atm, dtype=cp.int32)
        self.bas = cp.asarray(cell._bas, dtype=cp.int32)
        self.env = cp.asarray(cell._env, dtype=cp.float64)

        self.neighboring_images = cp.asarray(neighboring_images, dtype=cp.double)

        log_precision = int(np.ceil(-np.log10(self.precision)))

        counts = cp.zeros(1, dtype=cp.int32)
        libaft.count_non_trivial_pairs(
            multigrid_v2.cast_to_pointer(counts),
            ctypes.c_int(self.n_primitives),
            multigrid_v2.cast_to_pointer(self.neighboring_images),
            ctypes.c_int(len(self.neighboring_images)),
            multigrid_v2.cast_to_pointer(self.atm),
            multigrid_v2.cast_to_pointer(self.bas),
            multigrid_v2.cast_to_pointer(self.env),
            ctypes.c_int(log_precision),
        )

        n_pairs = int(counts[0])
        primitive_pairs = cp.zeros(n_pairs, dtype=cp.int32)
        image_list = cp.zeros(n_pairs, dtype=cp.int32)
        cutoffs = cp.zeros(n_pairs, dtype=cp.float64)
        angulars = cp.zeros(n_pairs, dtype=cp.int32)

        counts *= 0
        libaft.screen_gaussian_pairs(
            multigrid_v2.cast_to_pointer(primitive_pairs),
            multigrid_v2.cast_to_pointer(image_list),
            multigrid_v2.cast_to_pointer(angulars),
            multigrid_v2.cast_to_pointer(cutoffs),
            multigrid_v2.cast_to_pointer(counts),
            ctypes.c_int(self.n_primitives),
            multigrid_v2.cast_to_pointer(self.neighboring_images),
            ctypes.c_int(len(self.neighboring_images)),
            multigrid_v2.cast_to_pointer(self.atm),
            multigrid_v2.cast_to_pointer(self.bas),
            multigrid_v2.cast_to_pointer(self.env),
            ctypes.c_int(log_precision),
        )

        left_angular = angulars // 10
        right_angular = angulars - left_angular * 10
        angular_pair = cp.array([left_angular, right_angular], dtype=cp.int32).T

        unique_angular_pairs, unique_indices = multigrid_v2.unique_with_multiple_keys(angular_pair)
        unique_angular_pairs = unique_angular_pairs.get()

        self.screened_pairs = []
        self.n_blocks = self.mesh // 8 * 2

        for i, angular in enumerate(unique_angular_pairs):
            corresponding_pairs = cp.where(unique_indices == i)[0]

            cutoff = cutoffs[corresponding_pairs]
            sort_index = cp.argsort(-cutoff)

            corresponding_pairs = corresponding_pairs[sort_index]
            cutoff = cutoff[sort_index]
            shells = primitive_pairs[corresponding_pairs]
            image = image_list[corresponding_pairs]
            n_pairs_on_blocks = cp.zeros(np.prod(self.n_blocks) + 1, dtype=cp.int32)
            libaft.count_pairs_on_blocks(
                multigrid_v2.cast_to_pointer(n_pairs_on_blocks),
                multigrid_v2.cast_to_pointer(cutoff),
                ctypes.c_int(len(cutoff)),
                ctypes.c_int(self.mesh[0]),
                ctypes.c_int(self.mesh[1]),
                ctypes.c_int(self.mesh[2]),
                ctypes.c_int(self.n_blocks[0]),
                ctypes.c_int(self.n_blocks[1]),
                ctypes.c_int(self.n_blocks[2]),
            )

            sorted_block_indices = cp.argsort(-n_pairs_on_blocks)[: n_pairs_on_blocks[-1]]
            n_pairs_on_blocks = n_pairs_on_blocks[:-1]

            self.screened_pairs.append(
                {
                    'angular': angular,
                    'images': image,
                    'cutoffs': cutoff,
                    'shells': shells,
                    'n_pairs_per_block': cp.full(np.prod(self.n_blocks), len(cutoff), dtype=cp.int32),
                    'sorted_block_indices': cp.arange(np.prod(self.n_blocks), dtype=cp.int32),
                }
            )

        return self

    def get_j(self, dm: cp.ndarray, hermi=1, kpts=None, kpts_band=None, omega=None, exxdiv='ewald'):
        if kpts is not None:
            raise NotImplementedError
        pass
        # vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        # return vj

    def evaluate_reciprocal_density(self, dm: cp.ndarray):
        cell = self.primitive
        dm_shape = dm.shape
        n_ao = self.n_primitive_functions
        dm_primitive_shape = (n_ao, n_ao)
        n_k_points = 1
        n_grid_points = len(self.grid.coords)
        dm = dm.reshape(-1, *dm_shape[-2:])
        dm_primitive = multigrid_v2.contracted_to_primitive(
            dm, self.transform_coeff, self.transform_coeff, dm_primitive_shape
        )

        result = cp.zeros((n_k_points, n_grid_points), dtype=cp.complex128)

        for k in range(n_k_points):
            for pairs in self.screened_pairs:
                libaft.evaluate_density(
                    multigrid_v2.cast_to_pointer(result[k]),
                    multigrid_v2.cast_to_pointer(dm_primitive[k]),
                    multigrid_v2.cast_to_pointer(pairs['shells']),
                    ctypes.c_int(self.n_primitives),
                    multigrid_v2.cast_to_pointer(pairs['n_pairs_per_block']),
                    multigrid_v2.cast_to_pointer(self.shell_to_ao),
                    ctypes.c_int(self.n_primitive_functions),
                    multigrid_v2.cast_to_pointer(pairs['sorted_block_indices']),
                    ctypes.c_int(len(pairs['sorted_block_indices'])),
                    multigrid_v2.cast_to_pointer(pairs['images']),
                    multigrid_v2.cast_to_pointer(self.neighboring_images),
                    ctypes.c_int(len(self.neighboring_images)),
                    ctypes.c_int(self.mesh[0]),
                    ctypes.c_int(self.mesh[1]),
                    ctypes.c_int(self.mesh[2]),
                    ctypes.c_int(self.n_blocks[0]),
                    ctypes.c_int(self.n_blocks[1]),
                    ctypes.c_int(self.n_blocks[2]),
                    multigrid_v2.cast_to_pointer(self.atm),
                    multigrid_v2.cast_to_pointer(self.bas),
                    multigrid_v2.cast_to_pointer(self.env),
                    ctypes.c_int(pairs['angular'][0]),
                    ctypes.c_int(pairs['angular'][1]),
                )

        return result

    get_vxc = nr_vxc = NotImplemented  # numint_cpu.KNumInt.nr_vxc

    eval_xc_eff = mol_numint.eval_xc_eff
    _init_xcfuns = pbc_numint.NumInt._init_xcfuns

    nr_rks_fxc = NotImplemented
    nr_uks_fxc = NotImplemented
    nr_rks_fxc_st = NotImplemented
    cache_xc_kernel = NotImplemented
    cache_xc_kernel1 = NotImplemented

    to_gpu = utils.to_gpu
    device = utils.device

    def to_cpu(self):
        raise RuntimeError('Not available')


cell = gto.Cell(
    a=np.eye(3) * 3.5668,
    atom="""O     0.      0.      0.    
            H     0.8917  0.8917  0.8917
            H     1.7834  1.7834  0.    
         """,
    basis='gth-dzvp',
    pseudo='gth-pbe',
    precision=1e-8,
    verbose=5,
)
cell.build()
import pyscf

# cell = super_cell(cell, [3, 3, 3])

# mf = RKS(cell, xc='pbe')
exp_numint = AFTDFNumInt(cell)
# dm = cp.ones((cell.nao_nr(), cell.nao_nr()))
# exp_density = exp_numint.evaluate_reciprocal_density(dm)
# print(exp_density)
# ref_numint = multigrid_v2.MultiGridNumInt(cell)
# ref_density = multigrid_v2.evaluate_density_on_g_mesh(ref_numint, dm)
# print(cp.abs(exp_density - ref_density).max())
