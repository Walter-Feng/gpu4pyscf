from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from gpu4pyscf.pbc import dft as pbcdft
from gpu4pyscf.pbc.dft import multi_grid

from pyscf.pbc import gto
from pyscf.pbc import dft as cpudft
from pyscf.pbc.dft import multigrid as cpu_multi_grid

diamond_cell = bulk('C', 'diamond', orthorhombic=True, a=4)

lattice_vectors = diamond_cell.cell
cell = gto.M(
    h=np.array(diamond_cell.cell),
    atom=ase_atoms_to_pyscf(diamond_cell),
    basis='gth-tzv2p',
    verbose=6,
    unit='aa',
    ke_cutoff=200,
    pseudo = 'gth-pade'
)
cell.exp_to_discard = 0.1

cell = tools.super_cell(cell, [2, 2, 2])

gpu_mf = pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.max_cycle = 3
gpu_mf.init_guess = 'atom' # atom guess is fast
gpu_mf = multi_grid.fftdf(gpu_mf)
gpu_mf.with_df.ngrids = 4  # number of sets of grid points
gpu_mf.kernel()

mf=cpudft.RKS(cell)
mf.xc = "LDA"
# mf.xc = "PBE,PBE"
mf.max_cycle = 3
mf.init_guess = 'atom' # atom guess is fast
mf.with_df = cpu_multi_grid.MultiGridFFTDF(cell)
mf.with_df.ngrids = 4 # number of sets of grid points
mf.kernel()
