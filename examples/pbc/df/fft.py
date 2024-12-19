from ase.build import bulk
from gpu4pyscf.pbc.scf.khf import RHF, KRHF
from pyscf.pbc import gto
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np

cell = gto.M(
    h=np.eye(3) * 3.5668,
    atom= ase_atoms_to_pyscf(bulk('C', 'diamond', a=3.5668)),
    basis='gth-dzvp-molopt-sr',
    pseudo='gth-pade',
    verbose=5,
    unit = 'aa',
    ke_cutoff = 200
)
cell.exp_to_discard = 0.1
cell.max_memory = 64000

mf = RHF(cell)
mf.scf()

mp_grid = np.array([2, 2, 2])  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(mp_grid)
mf_with_k = KRHF(cell, kpts)

mf_with_k.scf()

