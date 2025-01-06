import cupy.cuda
from ase.build import bulk
from pyscf.pbc import gto, tools
from gpu4pyscf.pbc import dft
from pyscf.pbc.dft import multigrid
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np

with cupy.cuda.Device(1):
    cell = gto.M(
        h=np.eye(3) * 3.5668,
        atom= ase_atoms_to_pyscf(bulk('C', 'diamond', a=3.5668)),
        basis='gth-dzvp-molopt-sr',
        pseudo='gth-pade',
        verbose=6,
        unit = 'aa',
        ke_cutoff = 200
    )
    cell.exp_to_discard = 0.1
    cell.max_memory = 64000
    cell = tools.super_cell(cell, [4, 4, 4])

    mf = dft.KRKS(cell)
    mf.xc = 'lda,vwn'

    kpts = cell.make_kpts([1, 1, 1])
    mf = dft.KRKS(cell, kpts)
    mf.xc = 'lda,vwn'
    mf.with_df = multigrid.MultiGridFFTDF2(cell, kpts)
    mf.kernel()


# mf = RHF(cell)
# mf.scf()
#
# mp_grid = np.array([2, 2, 2])  # 4 k-points for each axis, 4^3=64 kpts in total
# kpts = cell.make_kpts(mp_grid)
# mf_with_k = KRHF(cell, kpts)
#
# mf_with_k.scf()

