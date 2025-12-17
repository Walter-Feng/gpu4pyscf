#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
        self.shell_to_ao = cp.asarray(make_loc(cell._bas, 'cart'), dtype=cp.int32)
        self.n_primitive_functions = transform_shape[0]
        self.n_functions = transform_shape[1]

        self.primitive_values = pbc_numint.eval_ao(cell, self.grid.coords)

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
