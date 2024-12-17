import gpu4pyscf.pbc.df.fft
from ase.build import bulk
from gpu4pyscf.pbc.scf.khf import KRHF
from gpu4pyscf.pbc.df.fft import FFTDF
# from pyscf.pbc.scf.khf import KRHF
# from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.scf.khf import KRHF as cpu_KRHF
from pyscf.pbc.df.fft import FFTDF as cpu_FFTDF
from pyscf.pbc import gto, scf, dft
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
import cupy

cell = gto.M(
    h=np.eye(3) * 3.5668,
    atom= ase_atoms_to_pyscf(bulk('C', 'diamond', a=3.5668)),
    basis='sto-3g',
    pseudo='gth-pade',
    verbose=5,
    unit = 'aa',
    ke_cutoff = 200
)
cell.exp_to_discard = 0.1
cell.max_memory = 64000

mp_grid = np.array([1, 1, 1])  # 4 k-points for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(mp_grid)
mf = KRHF(cell, kpts)
cpu_mf = cpu_KRHF(cell, kpts)
cpu_mf.with_df = cpu_FFTDF(cell, kpts)

mf.scf()

# dm = cpu_mf.make_rdm1()
#
# print("gpu total energy: ", mf.energy_tot(cupy.asarray(dm)))
# print("cpu total energy: ", cpu_mf.energy_tot(dm))
