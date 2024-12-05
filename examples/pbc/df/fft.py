from gpu4pyscf.pbc.scf.rhf import KRHF
from gpu4pyscf.pbc.df.fft import FFTDF
from pyscf.pbc import gto, scf, dft
import numpy as np
import cupy

with cupy.cuda.Device(4):

    cell = gto.M(
        h=np.eye(3) * 3.5668,
        atom='''C 0 0 0
                O 1 0 0
                O -1 0 0''',
        basis='gth-szv',
        pseudo='gth-pade',
        mesh=[20] * 3,
        verbose=6,
    )

    mp_grid = np.array([1, 1, 1])  # 4 k-points for each axis, 4^3=64 kpts in total
    kpts = cell.make_kpts(mp_grid)
    mf = KRHF(cell, kpts)
    mf.with_df = FFTDF(cell, kpts)
    mf.kernel()
