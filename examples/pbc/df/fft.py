import gpu4pyscf.pbc.df.fft
from ase.build import bulk
from gpu4pyscf.pbc.scf.rhf import KRHF
from gpu4pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
import cupy

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

mp_grid = np.array([2, 2, 2])  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(mp_grid)
mf = KRHF(cell, kpts)
mf.with_df = FFTDF(cell, kpts)
mf.max_cycle = 1
mf.kernel()
