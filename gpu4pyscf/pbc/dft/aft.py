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

libaft = load_library('libaft')


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
    verbose=0,
)
cell.build()
import pyscf

cell = pyscf.pbc.tools.super_cell(cell, [3, 3, 3])

mf = RKS(cell, xc='pbe')
exp_numint = AFTDFNumInt(cell)
dm = cp.ones((cell.nao_nr(), cell.nao_nr()))
exp_density = exp_numint.evaluate_reciprocal_density(dm)
print(exp_density)
ref_numint = multigrid_v2.MultiGridNumInt(cell)
ref_density = multigrid_v2.evaluate_density_on_g_mesh(ref_numint, dm)
print(cp.abs(exp_density - ref_density).max())
