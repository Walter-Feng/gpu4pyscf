import numpy
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid as cpu_multi_grid
from pyscf.pbc.grad import rhf as cpu_grad

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.dft import multigrid as gpu_multi_grid_qiming
import cupy as cp
import numpy as np
import scipy.optimize

cell = gto.Cell()

import cupy.cuda

# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
boxlen = 12
cell.a = numpy.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
cell.atom = """
O 0 0 0
H 0 0 1
H 0 1 0
"""

cell.basis = "sto-3g"
cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
cell.max_memory = 40000  # in MB
cell.precision = 1e-8  # integral precision
# cell.pseudo = "gth-pade"
cell.verbose = 5 
cell.use_loose_rcut = True  # integral screening based on shell radii
cell.use_particle_mesh_ewald = True  # use particle mesh ewald for nuclear repulsion
cell.build()

gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.init_guess = "atom"  # atom guess is fast
# gpu_mf.max_cycle = 0
# gpu_mf = gpu_multi_grid_mine.fftdf(gpu_mf)
# qiming_numint = gpu_multi_grid_qiming.MultiGridNumInt(cell)
# gpu_mf._numint = qiming_numint
# gpu_mf.with_df.ngrids = 4
# gpu_mf.kernel()

my_mf = gpu_multi_grid_mine.fftdf(gpu_mf)
my_mf.kernel()
converged_dm = my_mf.make_rdm1()
rhoG = gpu_multi_grid_mine.evaluate_density_on_g_mesh(my_mf.with_df, converged_dm, my_mf.kpts)
print("analytical: ")
print(my_mf.with_df.get_veff_ip1(converged_dm, my_mf.xc))
print(my_mf.with_df.get_veff_ip1(converged_dm, my_mf.xc).sum(axis=0))

cupy.cuda.stream.get_current_stream().synchronize()
def nr_rks_energy(coords):
    cell_with_new_coords = cell.set_geom_(coords.reshape(-1, 3), unit="Bohr", inplace=False)
    cell_with_new_coords.a = cell.lattice_vectors()
    fftdf = gpu_multi_grid_mine.FFTDF(cell_with_new_coords)
    _, e, _ = gpu_multi_grid_mine.nr_rks(fftdf, "LDA", converged_dm, with_j=False)
       
    return e 

print("numerical: ")
print(scipy.optimize.approx_fprime(cell.atom_coords().flatten(), nr_rks_energy, 1e-8).reshape(-1, 3))
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
"""