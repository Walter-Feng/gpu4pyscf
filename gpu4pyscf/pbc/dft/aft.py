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
from random import gauss
import numpy as np
import cupy as cp

import pyscf.pbc.gto as gto
from pyscf.pbc.gto.eval_gto import get_lattice_Ls
from pyscf.gto.moleintor import make_loc
from pyscf.gto import ATOM_OF, ANG_OF, NPRIM_OF, PTR_EXP, PTR_COEFF, PTR_COORD
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

libaft = load_library('libaft')

# s[i, n] := Solve[ Integrate[4 Pi g^2 g^i Exp[- g^2], {g, g0, Infinity}] == 10^(-n), g0]
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


# s[n] := Integrate[4 Pi g^2 g^n Exp[- g^2], {g, 0, Infinity}]
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


def update_reciprocal_lattice(cell):
    libaft.update_reciprocal_lattice_vectors(
        multigrid_v2.cast_to_pointer(cp.asarray(cell.reciprocal_vectors().T, order='C')),
    )


def check_cartesian(density, Gv, p, q, exponent, i_angular, j_angular):
    Gv = cp.asarray(Gv.T, order='C')
    n_grid = Gv.shape[-1]
    result = cp.zeros(n_grid, dtype=cp.complex128)

    libaft.check_cartesian(
        multigrid_v2.cast_to_pointer(result),
        multigrid_v2.cast_to_pointer(density),
        multigrid_v2.cast_to_pointer(Gv),
        ctypes.c_int(n_grid),
        ctypes.c_double(p[0]),
        ctypes.c_double(p[1]),
        ctypes.c_double(p[2]),
        ctypes.c_double(q[0]),
        ctypes.c_double(q[1]),
        ctypes.c_double(q[2]),
        ctypes.c_double(exponent),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
    )

    return result * common_fac_sp(i_angular) * common_fac_sp(j_angular)


def evaluate_density(
    density,
    non_trivial_pairs,
    n_shells,
    n_contributing_pairs_in_blocks,
    shell_to_ao_indices,
    n_functions,
    sorted_block_index,
    image_indices,
    vectors_to_neighboring_images,
    mesh,
    n_blocks,
    atm,
    bas,
    env,
    i_angular,
    j_angular,
):
    grid_size = np.prod(mesh)
    result = cp.zeros(grid_size, dtype=cp.complex128)

    libaft.evaluate_density(
        multigrid_v2.cast_to_pointer(result),
        multigrid_v2.cast_to_pointer(density),
        multigrid_v2.cast_to_pointer(non_trivial_pairs),
        ctypes.c_int(n_shells),
        multigrid_v2.cast_to_pointer(n_contributing_pairs_in_blocks),
        multigrid_v2.cast_to_pointer(shell_to_ao_indices),
        ctypes.c_int(n_functions),
        multigrid_v2.cast_to_pointer(sorted_block_index),
        ctypes.c_int(len(sorted_block_index)),
        multigrid_v2.cast_to_pointer(image_indices),
        multigrid_v2.cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(len(vectors_to_neighboring_images)),
        ctypes.c_int(mesh[0]),
        ctypes.c_int(mesh[1]),
        ctypes.c_int(mesh[2]),
        ctypes.c_int(n_blocks[0]),
        ctypes.c_int(n_blocks[1]),
        ctypes.c_int(n_blocks[2]),
        multigrid_v2.cast_to_pointer(atm),
        multigrid_v2.cast_to_pointer(bas),
        multigrid_v2.cast_to_pointer(env),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
    )

    return result


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
        update_reciprocal_lattice(cell)

        self.cell = cell
        if self.cell.precision:
            self.precision = self.cell.precision
        else:
            self.precision = 1e-8

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

        self.Gv = cp.asarray(cell.get_Gv())

        self.primitive_values = pbc_numint.eval_ao(cell, self.grid.coords)
        self.neighboring_images = cp.asarray(neighboring_images, dtype=cp.double)

        left_shell_list = []
        right_shell_list = []
        image_list = []
        angular_pair = []
        cutoffs = []

        log_precision = int(np.ceil(-np.log10(self.precision)))

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
                    * 2**total_angular  # leading hermitian polynomial factor
                )
                i_center_ptr, j_center_ptr = cell._atm[[i_atom_index, j_atom_index], PTR_COORD]
                i_center_reference = cell._env[i_center_ptr : i_center_ptr + 3]
                j_center = cell._env[j_center_ptr : j_center_ptr + 3]
                integration_factor = gaussian_integration_values[total_angular]

                neighboring_images_on_host = self.neighboring_images.get()
                for image_index in range(len(self.neighboring_images)):
                    i_center = i_center_reference + neighboring_images_on_host[image_index]
                    pair_center = (i_exponent * i_center + j_exponent * j_center) / pair_exponent
                    from_i = pair_center - i_center
                    from_j = pair_center - j_center
                    exponent_prefactor = np.exp(
                        -i_exponent * j_exponent / pair_exponent * np.sum((from_i - from_j) ** 2)
                    )

                    total_prefactor = pair_prefactor * exponent_prefactor * integration_factor
                    if total_prefactor < self.precision:
                        continue
                    else:
                        left_shell_list.append(i_primitive)
                        right_shell_list.append(j_primitive)
                        image_list.append(image_index)
                        angular_pair.append((i_angular, j_angular))

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

        left_shell_list = cp.array(left_shell_list, dtype=cp.int32)
        right_shell_list = cp.array(right_shell_list, dtype=cp.int32)
        image_list = cp.array(image_list, dtype=cp.int32)
        cutoffs = cp.array(cutoffs)
        angular_pair = cp.array(angular_pair, dtype=cp.int32)

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

            left = left_shell_list[corresponding_pairs]
            right = right_shell_list[corresponding_pairs]
            image = image_list[corresponding_pairs]
            self.screened_pairs.append(
                {
                    'angular': angular,
                    'left_shell': left,
                    'right_shell': right,
                    'images': image,
                    'cutoffs': cutoff,
                    'shells': left * self.n_primitives + right,
                    'n_pairs_per_block': cp.full(np.prod(self.n_blocks), len(cutoff), dtype=cp.int32),
                }
            )

        self.atm = cp.asarray(cell._atm, dtype=cp.int32)
        self.bas = cp.asarray(cell._bas, dtype=cp.int32)
        self.env = cp.asarray(cell._env, dtype=cp.float64)

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

        for pairs in self.screened_pairs:
            density = evaluate_density(
                dm_primitive,
                pairs['shells'],
                self.n_primitives,
                pairs['n_pairs_per_block'],
                self.shell_to_ao,
                self.n_primitive_functions,
                cp.arange(np.prod(self.n_blocks), dtype=cp.int32),
                pairs['images'],
                self.neighboring_images,
                self.mesh,
                self.n_blocks,
                self.atm,
                self.bas,
                self.env,
                *pairs['angular'],
            )

            result[0] += density

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
    precision=1e-10,
    verbose=0,
)

mf = RKS(cell, xc='pbe')
exp_numint = AFTDFNumInt(cell)
exp_numint.build()
dm = cp.ones((cell.nao_nr(), cell.nao_nr()))
exp_density = exp_numint.evaluate_reciprocal_density(dm)
print(exp_density)
ref_numint = multigrid_v2.MultiGridNumInt(cell)
ref_density = multigrid_v2.evaluate_density_on_g_mesh(ref_numint, dm)
print(cp.abs(exp_density - ref_density).max())
