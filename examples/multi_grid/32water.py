import numpy
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid as cpu_multi_grid
from pyscf.pbc.grad import rhf as cpu_grad

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.dft import multigrid as gpu_multi_grid_qiming
import cupy as cp
cell = gto.Cell()

import cupy.cuda

# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
boxlen = 12
cell.a = numpy.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
cell.atom = """
O   1.613159999999999927e+00 6.554399000000000086e+00 4.322530000000000427e+00
 H   1.428539999999999921e+00 5.878300000000000303e+00 5.021608999999999767e+00
 H   2.016700000000000159e+00 5.965879000000000154e+00 3.618549999999999933e+00
 O   3.162049999999999805e+00 8.750220000000000553e+00 4.021650000000000169e+00
 H   2.443400000000000016e+00 8.029849999999999710e+00 4.224420000000000286e+00
 H   3.217699999999999783e+00 9.223639999999999617e+00 4.884699999999999598e+00
"""

cell.basis = "gth-tzv2p"
cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
cell.max_memory = 40000  # in MB
cell.precision = 1e-8  # integral precision
cell.pseudo = "gth-pade"
cell.verbose = 3
cell.use_loose_rcut = True  # integral screening based on shell radii
cell.use_particle_mesh_ewald = True  # use particle mesh ewald for nuclear repulsion
cell.build()

print("=" * 100)
print("gpu_multi_grid_mine")
print("=" * 100)
gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.init_guess = "atom"  # atom guess is fast
# gpu_mf.max_cycle = 0
gpu_mf = gpu_multi_grid_mine.fftdf(gpu_mf)
gpu_mf.with_df.ngrids = 4
gpu_mf.kernel()
converged_dm = gpu_mf.make_rdm1()

print(gpu_mf.with_df.get_veff_ip1(gpu_mf.make_rdm1(), gpu_mf.xc))
print(gpu_mf.with_df.get_veff_ip1(gpu_mf.make_rdm1(), gpu_mf.xc).sum(axis=0))

""" 
print("="*100)
print("gpu_multi_grid_qiming")
print("="*100)
gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "PBE"
gpu_mf.init_guess = 'atom'
# gpu_mf.max_cycle = 0
gpu_mf._numint = gpu_multi_grid_qiming.MultiGridNumInt(cell)
gpu_mf.kernel()
 """
print("=" * 100)
print("cpu_multi_grid")
print("=" * 100)
mf = dft.RKS(cell)
mf.xc = "LDA"
# mf.xc = "PBE,PBE"
# mf.max_cycle = 0
# mf.init_guess = 'atom' # atom guess is fast
mf.with_df = cpu_multi_grid.MultiGridFFTDF2(cell)
mf.with_df.ngrids = 4  # number of sets of grid points
mf.kernel()
gradient = cpu_grad.Gradients(mf)
gradient.grad_elec()

