from ase.build import bulk
from gpu4pyscf.pbc.scf.khf import KRHF
from pyscf.pbc import gto, tools
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
import cupy

from gpu4pyscf import mpi

# mpi.comm = mpi.Communicator([2, 3])

if mpi.comm.is_main:
    verbose = 6
else:
    verbose = 0

diamond_cell = bulk('C', 'diamond', a=3.5668)

lattice_vectors = diamond_cell.cell
cell = gto.M(
    h=np.array(lattice_vectors),
    atom=ase_atoms_to_pyscf(bulk('C', 'diamond', a=3.5668)),
    basis='gth-dzvp-molopt-sr',
    pseudo='gth-pade',
    verbose=verbose,
    unit='aa',
    ke_cutoff=200
)
cell.exp_to_discard = 0.1

cell = tools.super_cell(cell, [10, 1, 1])

mp_grid = np.array([1, 1, 1])  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(mp_grid)
mf_with_k = KRHF(cell, kpts)
mf_with_k.max_cycle = 1
mf_with_k.with_df.block_size = 4

mf_with_k.scf()
