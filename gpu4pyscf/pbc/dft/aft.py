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


import numpy as np
import cupy as cp

import pyscf.pbc.gto as gto
from pyscf.pbc.gto.eval_gto import get_lattice_Ls
from pyscf.gto.moleintor import make_loc
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, PTR_EXP, PTR_COEFF, PTR_COORD
from pyscf import lib

from gpu4pyscf.dft import numint as mol_numint
from gpu4pyscf.pbc.dft import RKS
from gpu4pyscf.pbc.dft import numint as pbc_numint
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.pbc.tools import pbc as pbc_tools
import gpu4pyscf.pbc.dft.multigrid_v2 as multigrid_v2
import gpu4pyscf.pbc.dft.gen_grid as gen_grid
from gpu4pyscf.lib.cupy_helper import contract, tag_array, load_library


def common_fac_sp(angular: int) -> float:
    match angular:
        case 0:
            return 0.282094791773878143
        case 1:
            return 0.488602511902919921
        case _:
            return 1.0


def polynomials(coords: cp.ndarray, angular: int) -> cp.ndarray:
    assert coords.shape[-1] == 3
    x, y, z = coords.T
    n_functions = (angular + 1) * (angular + 2) // 2

    result = cp.ones((n_functions, len(x)))

    match angular:
        case 0:
            pass
        case 1:
            result[0] = x
            result[1] = y
            result[2] = z
        case 2:
            result[0] = x * x
            result[1] = x * y
            result[2] = x * z
            result[3] = y * y
            result[4] = y * z
            result[5] = z * z
        case _:
            raise NotImplementedError

    return result * common_fac_sp(angular)


def orbital_coefficient(angular: int) -> cp.ndarray:
    match angular:
        case 0:
            return cp.ones((1, 3, 1)) * common_fac_sp(angular)
        case 1:
            pass


def hermite_polynomials(g: cp.ndarray, max_power: int) -> cp.ndarray:
    result = cp.ones((max_power + 1, len(g)))

    if max_power == 0:
        return result

    result[1] = g

    for i in range(max_power - 1):
        result[i + 2] = 2 * g * result[i + 1] - 2 * (i + 1) * result[i]

    return result


def hermite_polynomials_in_xyz(g_xyz: cp.ndarray, max_power: int) -> cp.ndarray:
    result = cp.ones((3, (max_power + 1), g_xyz.shape[-1]))

    for i in range(3):
        result[i] = hermite_polynomials(g_xyz[i], max_power)

    return result


def binomial_tensor(to_i: float, to_j: float, i_angular: int, j_angular: int):
    dd_polynomial_tensor = cp.array(
        [
            [[1, 0, 0, 0, 0], [to_j, 1, 0, 0, 0], [to_j**2, 2 * to_j, 1, 0, 0]],
            [
                [to_i, 1, 0, 0, 0],
                [to_i * to_j, to_i + to_j, 1, 0, 0],
                [to_i * to_j**2, 2 * to_i * to_j + to_j**2, to_i + 2 * to_j, 1, 0],
            ],
            [
                [to_i**2, 2 * to_i, 1, 0, 0],
                [to_i**2 * to_j, to_i**2 + 2 * to_i * to_j, 2 * to_i + to_j, 1, 0],
                [
                    to_i**2 * to_j**2,
                    2 * to_i**2 * to_j + 2 * to_i * to_j**2,
                    to_i**2 + 4 * to_i * to_j + to_j**2,
                    2 * to_i + 2 * to_j,
                    1,
                ],
            ],
        ]
    )

    return dd_polynomial_tensor[: (i_angular + 1), : (j_angular + 1), (i_angular + j_angular + 1)]


class AFTDFNumInt(pbc_numint.NumInt):
    def __init__(self, cell: gto.Cell):
        self.cell = cell
        self.mesh = cell.mesh
        self.grid = gen_grid.UniformGrids(cell)
        self.tasks = None
        self.sorted_gaussian_pairs = None
        Gv = pbc_tools._get_Gv(cell, cell.mesh)
        self.coulG = pbc_tools.get_coulG(cell, Gv=Gv)
        self.build()

    def build(self):
        cell = self.cell
        cell, transform_coeff = cell.decontract_basis(to_cart=True)
        sorted_transform_coeff, transform_shape = multigrid_v2.sort_contraction_coefficients(transform_coeff)

        neighboring_images = get_lattice_Ls(cell)
        images_sort_index = np.argsort(np.linalg.norm(neighboring_images, axis=1))
        neighboring_images = neighboring_images[images_sort_index]

        self.primitive = cell
        self.transform_coeff = sorted_transform_coeff
        self.shell_to_ao = np.asarray(make_loc(cell._bas, 'cart'), dtype=cp.int32)
        self.n_primitives = len(cell._bas)
        self.n_primitive_functions = transform_shape[0]
        self.n_functions = transform_shape[1]

        self.Gv = cp.asarray(cell.get_Gv())

        self.primitive_values = pbc_numint.eval_ao(cell, self.grid.coords)
        self.neighboring_images = neighboring_images

        exponents = cell._env[cell._bas[:, PTR_EXP]]
        angulars = cell._bas[:, ANG_OF]
        atom_indices = cell._bas[:, ATOM_OF]
        x = cell._env[cell._atm[atom_indices, PTR_COORD]]
        y = cell._env[cell._atm[atom_indices, PTR_COORD] + 1]
        z = cell._env[cell._atm[atom_indices, PTR_COORD] + 2]
        primitive_coords = np.transpose(np.array([x, y, z]))

        return self

    def get_j(self, dm: cp.ndarray, hermi=1, kpts=None, kpts_band=None, omega=None, exxdiv='ewald'):
        if kpts is not None:
            raise NotImplementedError
        pass
        # vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        # return vj

    def get_primitive_pair_values_in_reciprocal(self, i_primitive, j_primitive):
        primitive_pair = [i_primitive, j_primitive]
        real_coords = cp.asarray(self.grid.coords)
        cell = self.primitive

        i_function_indices = np.arange(self.shell_to_ao[i_primitive], self.shell_to_ao[i_primitive + 1])
        j_function_indices = np.arange(self.shell_to_ao[j_primitive], self.shell_to_ao[j_primitive + 1])

        i_atom_index, j_atom_index = cell._bas[primitive_pair, ATOM_OF]
        i_angular, j_angular = cell._bas[primitive_pair, ANG_OF]

        i_exponent, j_exponent = cell._env[cell._bas[primitive_pair, PTR_EXP]]
        i_coeff, j_coeff = cell._env[cell._bas[primitive_pair, PTR_COEFF]]

        i_center_ptr, j_center_ptr = cell._atm[[i_atom_index, j_atom_index], PTR_COORD]
        i_center_reference = cell._env[i_center_ptr : i_center_ptr + 3]
        j_center_reference = cell._env[j_center_ptr : j_center_ptr + 3]

        result = cp.zeros((len(i_function_indices), len(j_function_indices), np.prod(self.cell.mesh)))
        i_function = cp.zeros((len(i_function_indices), np.prod(self.cell.mesh)))
        j_function = cp.zeros((len(j_function_indices), np.prod(self.cell.mesh)))

        for image in self.neighboring_images:
            i_center = cp.asarray(i_center_reference + image)
            i_polynomial = polynomials(real_coords - i_center, i_angular)
            i_exp = i_coeff * cp.exp(-i_exponent * cp.linalg.norm(real_coords - i_center, axis=1) ** 2)
            i_result = i_polynomial * i_exp
            i_function += i_result

            j_center = cp.asarray(j_center_reference + image)
            j_polynomial = polynomials(real_coords - j_center, j_angular)
            j_exp = j_coeff * cp.exp(-j_exponent * cp.linalg.norm(real_coords - j_center, axis=1) ** 2)
            j_result = j_polynomial * j_exp
            j_function += j_result

        result = cp.einsum('ag, bg-> abg', i_function, j_function)

        return result

    def evaluate_reciprocal_density(self, dm: cp.ndarray):
        cell = self.primitive
        dm_shape = dm.shape
        n_ao = self.n_primitive_functions
        dm_primitive_shape = (n_ao, n_ao)
        n_k_points = 1
        n_grid_points = len(self.grid.coords)
        weight = 1.0 / n_k_points * cell.vol / n_grid_points
        dm = dm.reshape(-1, *dm_shape[-2:]) * weight
        dm_primitive = multigrid_v2.contracted_to_primitive(
            dm, self.transform_coeff, self.transform_coeff, dm_primitive_shape
        )

        ao_pairs = cp.einsum('gi, gj -> ijg', self.primitive_values, self.primitive_values).reshape(-1, *cell.mesh)

        ao_pairs = multigrid_v2.fft_in_place(ao_pairs)
        ao_pairs = ao_pairs.reshape(*dm_primitive_shape, -1)
        return cp.einsum('kij, ijg -> kg', dm_primitive, ao_pairs)

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
    verbose=0,
)

mf = RKS(cell, xc='pbe')
exp_numint = AFTDFNumInt(cell)
dm = cp.ones((cell.nao_nr(), cell.nao_nr()))
ref_numint = multigrid_v2.MultiGridNumInt(cell)
grid_coords = exp_numint.grid.coords.get()
reciprocal_coords = cell.get_Gv()

hermite_G = hermite_polynomials_in_xyz(cp.asarray(reciprocal_coords.T), 5)
print(hermite_G)

# Note: analytical expression of a fft of one dimension gaussian function f(x)
# is equivalent to, in Mathematica,
# Sqrt[2Pi] InverseFourierTransform[f[x], x, G]
# So in three dimensional systems it should be something like, analytically,
# F[ x^a y^b z^c exp[- p (x^2 + y^2 + z^2)] =
#   (I 2 Sqrt[p])^(- a - b - c)
#           HermiteH[a, Gx / Sqrt[4 p]]
#           HermiteH[b, Gy / Sqrt[4 p]]
#           HermiteH[c, Gz / Sqrt[4 p]]
#               (Pi / p)^(3/2) exp [- (Gx^2 + Gy^2 + Gz^2) / 4 a]
# If centered at r0, should be good with multiplying with a phase factor exp[- I G r0].
# An example is given below.

alpha = 10.0
center = np.array([2.0, 1.0, 0.0])
grid_coords -= center
r = np.linalg.norm(grid_coords, axis=1).reshape(cell.mesh)
g = np.linalg.norm(reciprocal_coords, axis=1).reshape(cell.mesh)
test_gaussian = (
    np.exp(-alpha * r * r) * grid_coords[:, 0].reshape(cell.mesh) ** 3 * grid_coords[:, 1].reshape(cell.mesh)
)
ref_reciprocal = np.fft.fftn(test_gaussian, axes=(-3, -2, -1)) / len(grid_coords) * cell.vol
phase_factor = np.exp(-1j * reciprocal_coords @ center).reshape(cell.mesh)
exp_reciprocal = (
    np.exp(-g * g / 4 / alpha)
    * np.sqrt(np.pi / alpha) ** 3
    * 1j
    * (-6 * alpha * reciprocal_coords[:, 0].reshape(cell.mesh) + reciprocal_coords[:, 0].reshape(cell.mesh) ** 3)
    / (2 * alpha) ** 3
    * (-1j)
    * reciprocal_coords[:, 1].reshape(cell.mesh)
    / 2
    / alpha
) * phase_factor

diff = ref_reciprocal - exp_reciprocal
assert np.abs(diff).max() < 1e-10
