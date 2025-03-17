from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid as cpu_multi_grid

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.dft import multi_grid_store_ao as gpu_multi_grid_ao
from gpu4pyscf.pbc.dft import multigrid as gpu_multi_grid_qiming

cell=gto.Cell()

import cupy.cuda
# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
boxlen=12
diamond_cell = bulk('C', 'diamond', orthorhombic=True)

lattice_vectors = diamond_cell.cell
cell = gto.M(
    h=np.array(diamond_cell.cell),
    atom=ase_atoms_to_pyscf(diamond_cell),
    basis='gth-tzv2p',
    verbose=5,
    unit='aa',
    ke_cutoff=200,
    pseudo = 'gth-pade'
)
cell.exp_to_discard = 0.1

cell = tools.super_cell(cell, [1, 1, 1])
cell.build()

print("="*100)
print("gpu_multi_grid_mine")
print("="*100)
gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.init_guess = 'atom' # atom guess is fast
gpu_mf.max_cycle = 10
gpu_mf = gpu_multi_grid_mine.fftdf(gpu_mf)
gpu_mf.with_df.ngrids = 4
gpu_mf.kernel()

print("="*100)
print("gpu_multi_grid_qiming")
print("="*100)
gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.init_guess = 'atom'
gpu_mf.max_cycle = 10
gpu_mf._numint = gpu_multi_grid_qiming.MultiGridNumInt(cell)
gpu_mf.kernel()

print("="*100)
print("cpu_multi_grid")
print("="*100)  
mf=dft.RKS(cell)
mf.xc = "LDA"
# mf.xc = "PBE,PBE"
mf.max_cycle = 10
# mf.init_guess = 'atom' # atom guess is fast
mf.with_df = cpu_multi_grid.MultiGridFFTDF(cell)
mf.with_df.ngrids = 4 # number of sets of grid points
mf.kernel()

print("="*100)
print("gpu_multi_grid_ao")
print("="*100)
gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.init_guess = 'atom' # atom guess is fast
gpu_mf.max_cycle = 10
gpu_mf = gpu_multi_grid_ao.fftdf(gpu_mf)
gpu_mf.with_df.ngrids = 4
gpu_mf.kernel()


