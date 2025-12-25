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

gaussian_integration_cutoff_table = np.array(
    [
        [1.56458, 2.24417, 2.74084, 3.15292, 3.51327, 3.83771, 4.13534, 4.41192, 4.67142, 4.91669, 5.14986],
        [1.81533, 2.46995, 2.95244, 3.35429, 3.70657, 4.02436, 4.3163, 4.58795, 4.84308, 5.08443, 5.31406],
        [2.09092, 2.70638, 3.17011, 3.55956, 3.90255, 4.2129, 4.49865, 4.76499, 5.01549, 5.25274, 5.47867],
        [2.37682, 2.94857, 3.39132, 3.76715, 4.10012, 4.40256, 4.6818, 4.9426, 5.1883, 5.4213, 5.64344],
        [2.66372, 3.19284, 3.61404, 3.97578, 4.2984, 4.59269, 4.86524, 5.12038, 5.36117, 5.58986, 5.80815],
        [2.9466, 3.4366, 3.83673, 4.18444, 4.49665, 4.78274, 5.04855, 5.29798, 5.53383, 5.75818, 5.97261],
        [3.22305, 3.67817, 4.0583, 4.39234, 4.69431, 4.97228, 5.23138, 5.47513, 5.70606, 5.92608, 6.13664],
        [3.49208, 3.91647, 4.27794, 4.5989, 4.89094, 5.16095, 5.41346, 5.6516, 5.87766, 6.09339, 6.30012],
        [3.7535, 4.1509, 4.49513, 4.8037, 5.08618, 5.34848, 5.59455, 5.82719, 6.04847, 6.25997, 6.46293],
    ]
)

gaussian_integration_values = np.array(
    [5.56833, 6.28319, 8.35249, 12.5664, 20.8812, 37.6991, 73.0843, 150.796, 328.879]
)


def common_fac_sp(angular: int) -> float:
    match angular:
        case 0:
            return 0.282094791773878143
        case 1:
            return 0.488602511902919921
        case _:
            return 1.0


def power_series(coords: cp.ndarray, max_angular: int) -> cp.ndarray:
    assert coords.shape[-1] == 3

    result = cp.ones((3, max_angular + 1, len(coords)))

    for xyz in range(3):
        for i in range(max_angular):
            result[xyz, i + 1] = cp.power(coords[:, xyz], i + 1)

    return result


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

    result[1] = 2 * g

    for i in range(max_power - 1):
        result[i + 2] = 2 * g * result[i + 1] - 2 * (i + 1) * result[i]

    return result


def hermite_polynomials_in_xyz(g_xyz: cp.ndarray, max_power: int, exponent_reciprocal: float) -> cp.ndarray:
    result = cp.ones((3, (max_power + 1), g_xyz.shape[0]), dtype=cp.complex128)
    factor = cp.array([np.pow(-1j * np.sqrt(exponent_reciprocal), i) for i in range(max_power + 1)])

    for i in range(3):
        result[i] = factor[:, None] * hermite_polynomials(g_xyz[:, i] * np.sqrt(exponent_reciprocal), max_power)

    return result


def polynomial_tensor(angular: int):
    n_functions = (angular + 1) * (angular + 2) // 2

    result = np.zeros((n_functions, 3, angular + 1))
    result[:, :, 0] = 1.0
    match angular:
        case 0:
            pass
        case 1:
            result[0, 0] = np.array([0, 1])
            result[1, 1] = np.array([0, 1])
            result[2, 2] = np.array([0, 1])
        case 2:
            result[0, 0] = np.array([0, 0, 1])  # x^2

            result[1, 0] = np.array([0, 1, 0])  # x y
            result[1, 1] = np.array([0, 1, 0])  # x y

            result[2, 0] = np.array([0, 1, 0])  # x z
            result[2, 2] = np.array([0, 1, 0])  # x z

            result[3, 1] = np.array([0, 0, 1])  # y^2

            result[4, 1] = np.array([0, 1, 0])  # y z
            result[4, 2] = np.array([0, 1, 0])  # y z

            result[5, 2] = np.array([0, 0, 1])  # z^2
        case _:
            raise NotImplementedError

    return cp.asarray(result)


def binomial_tensor(from_i: float, from_j: float, i_angular: int, j_angular: int):
    # produces a C(i, j, k) tensor, which represents the coefficients
    # (x + x_i)^i (x+ x_j^j) = \sum C(i, j, k) x^k
    dd_polynomial_tensor = cp.array(
        [
            [[1, 0, 0, 0, 0], [from_j, 1, 0, 0, 0], [from_j**2, 2 * from_j, 1, 0, 0]],
            [
                [from_i, 1, 0, 0, 0],
                [from_i * from_j, from_i + from_j, 1, 0, 0],
                [from_i * from_j**2, 2 * from_i * from_j + from_j**2, from_i + 2 * from_j, 1, 0],
            ],
            [
                [from_i**2, 2 * from_i, 1, 0, 0],
                [from_i**2 * from_j, from_i**2 + 2 * from_i * from_j, 2 * from_i + from_j, 1, 0],
                [
                    from_i**2 * from_j**2,
                    2 * from_i**2 * from_j + 2 * from_i * from_j**2,
                    from_i**2 + 4 * from_i * from_j + from_j**2,
                    2 * from_i + 2 * from_j,
                    1,
                ],
            ],
        ]
    )

    return dd_polynomial_tensor[: (i_angular + 1), : (j_angular + 1), : (i_angular + j_angular + 1)]


def binomial_tensor_xyz(from_i: np.ndarray, from_j: np.ndarray, i_angular: int, j_angular: int):
    return cp.array([binomial_tensor(from_i[dim], from_j[dim], i_angular, j_angular) for dim in range(3)])


def polynomial_expansion(from_i: np.ndarray, from_j: np.ndarray, i_angular: int, j_angular: int):
    i_polynomial = polynomial_tensor(i_angular)
    j_polynomial = polynomial_tensor(j_angular)

    binomial_tensor = binomial_tensor_xyz(from_i, from_j, i_angular, j_angular)
    binomial_tensor *= np.pow(common_fac_sp(i_angular) * common_fac_sp(j_angular), 1 / 3)

    return cp.einsum('ixp, jxq, xpqc -> xijc', i_polynomial, j_polynomial, binomial_tensor)


def polynomial_expansion_on_coords(
    from_i: np.ndarray, from_j: np.ndarray, i_angular: int, j_angular: int, powered_coords: cp.ndarray
):
    expansion_tensor = polynomial_expansion(from_i, from_j, i_angular, j_angular)
    xyz_unmerged = cp.einsum('xijc, xcg->xijg', expansion_tensor, powered_coords)
    return cp.prod(xyz_unmerged, axis=0)


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

        left_shell_list = []
        right_shell_list = []
        image_list = []
        cutoffs = []

        precision = 1e-6
        log_precision = int(np.ceil(-np.log10(precision)))
        for i_primitive in range(self.n_primitives):
            for j_primitive in range(self.n_primitives):
                primitive_pair = [i_primitive, j_primitive]
                i_atom_index, j_atom_index = cell._bas[primitive_pair, ATOM_OF]
                i_angular, j_angular = cell._bas[primitive_pair, ANG_OF]
                total_angular = i_angular + j_angular

                i_exponent, j_exponent = cell._env[cell._bas[primitive_pair, PTR_EXP]]
                pair_exponent = i_exponent + j_exponent
                pair_exponent_reciprocal = 0.25 / pair_exponent

                i_coeff, j_coeff = cell._env[cell._bas[primitive_pair, PTR_COEFF]]
                pair_prefactor = (
                    i_coeff
                    * j_coeff
                    * np.sqrt((np.pi / pair_exponent) ** 3)
                    * np.sqrt(pair_exponent_reciprocal) ** (total_angular + 2)
                )
                i_center_ptr, j_center_ptr = cell._atm[[i_atom_index, j_atom_index], PTR_COORD]
                i_center_reference = cell._env[i_center_ptr : i_center_ptr + 3]
                j_center = cell._env[j_center_ptr : j_center_ptr + 3]
                integration_factor = gaussian_integration_values[total_angular]

                for image in range(len(self.neighboring_images)):
                    i_center = i_center_reference + image
                    pair_center = (i_exponent * i_center + j_exponent * j_center) / pair_exponent
                    from_i = pair_center - i_center
                    from_j = pair_center - j_center
                    exponent_prefactor = np.exp(
                        -i_exponent * j_exponent / pair_exponent * np.sum((from_i - from_j) ** 2)
                    )

                    total_prefactor = pair_prefactor * exponent_prefactor
                    if total_prefactor * integration_factor < precision:
                        continue
                    else:
                        left_shell_list.append(i_primitive)
                        right_shell_list.append(j_primitive)
                        image_list.append(image)

                        equivalent_offset_for_cutoff = int(np.ceil(np.log10(total_prefactor)))
                        offset = log_precision + equivalent_offset_for_cutoff
                        if offset < 0:
                            offset = 0
                        if offset > 9:
                            offset = 9

                        cutoff = gaussian_integration_cutoff_table[total_angular, offset] / np.sqrt(
                            pair_exponent_reciprocal
                        )

                        cutoffs.append(cutoff)

        left_shell_list = np.array(left_shell_list, dtype=cp.int32)
        right_shell_list = np.array(right_shell_list, dtype=cp.int32)
        image_list = np.array(image_list, dtype=cp.int32)
        cutoffs = np.array(cutoffs)

        sort_index = np.argsort(-cutoffs)
        left_shell_list = left_shell_list[sort_index]
        right_shell_list = right_shell_list[sort_index]
        image_list = image_list[sort_index]
        cutoffs = cutoffs[sort_index]

        return self

    def get_j(self, dm: cp.ndarray, hermi=1, kpts=None, kpts_band=None, omega=None, exxdiv='ewald'):
        if kpts is not None:
            raise NotImplementedError
        pass
        # vj = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        # return vj

    def get_primitive_pair_values_in_reciprocal(self, i_primitive, j_primitive, image_index, cutoff):
        real_coords = cp.asarray(self.grid.coords)
        cell = self.primitive

        i_function_indices = np.arange(self.shell_to_ao[i_primitive], self.shell_to_ao[i_primitive + 1])
        j_function_indices = np.arange(self.shell_to_ao[j_primitive], self.shell_to_ao[j_primitive + 1])

        primitive_pair = [i_primitive, j_primitive]
        i_atom_index, j_atom_index = cell._bas[primitive_pair, ATOM_OF]
        i_angular, j_angular = cell._bas[primitive_pair, ANG_OF]

        i_exponent, j_exponent = cell._env[cell._bas[primitive_pair, PTR_EXP]]
        pair_exponent = i_exponent + j_exponent
        pair_exponent_reciprocal = 0.25 / pair_exponent

        i_coeff, j_coeff = cell._env[cell._bas[primitive_pair, PTR_COEFF]]
        factor = i_coeff * j_coeff * np.sqrt((np.pi / pair_exponent) ** 3)

        i_center_ptr, j_center_ptr = cell._atm[[i_atom_index, j_atom_index], PTR_COORD]
        i_center_reference = cell._env[i_center_ptr : i_center_ptr + 3]
        j_center = cell._env[j_center_ptr : j_center_ptr + 3]

        screening = cp.ones(self.Gv.shape[0])
        screening[cp.linalg.norm(self.Gv, axis=1) > cutoff] = 0

        hermite_G = hermite_polynomials_in_xyz(self.Gv, i_angular + j_angular, pair_exponent_reciprocal)
        norm_squared = cp.sum(self.Gv * self.Gv, axis=1)

        image = self.neighboring_images[image_index]
        i_center = i_center_reference + image
        pair_center = (i_exponent * i_center + j_exponent * j_center) / pair_exponent
        from_i = pair_center - i_center
        from_j = pair_center - j_center
        exponent_prefactor = np.exp(-i_exponent * j_exponent / pair_exponent * np.sum((from_i - from_j) ** 2))

        cartesian = polynomial_expansion_on_coords(from_i, from_j, i_angular, j_angular, hermite_G)
        exp = cp.exp(-norm_squared * pair_exponent_reciprocal)
        exp *= screening
        phase = cp.exp(-1j * self.Gv @ cp.asarray(pair_center))

        return exp * phase * (factor * exponent_prefactor) * cartesian, (i_function_indices, j_function_indices)

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
