import numpy
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid as cpu_multi_grid

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.dft import multi_grid_store_ao as gpu_multi_grid_ao
from gpu4pyscf.pbc.dft import multigrid as gpu_multi_grid_qiming
cell=gto.Cell()

import cupy.cuda
# cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
boxlen=10
cell.a=numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom="""
 O   1.613159999999999927e+00 6.554399000000000086e+00 4.322530000000000427e+00
 H   1.428539999999999921e+00 5.878300000000000303e+00 5.021608999999999767e+00
 H   2.016700000000000159e+00 5.965879000000000154e+00 3.618549999999999933e+00
 O   3.162049999999999805e+00 8.750220000000000553e+00 4.021650000000000169e+00
 H   2.443400000000000016e+00 8.029849999999999710e+00 4.224420000000000286e+00
 H   3.217699999999999783e+00 9.223639999999999617e+00 4.884699999999999598e+00
 O   3.699879999999999836e+00 2.308800000000000019e-01 1.688099999999999934e+00
 H   3.378330000000000055e+00 -1.526800000000010094e-01 2.487979999999999858e+00
 H   3.611790000000000056e+00 -4.963400000000008916e-01 9.904600000000000071e-01
 O   2.761179999999999968e+00 4.815249999999999808e+00 2.635450000000000070e+00
 H   3.601900000000000102e+00 4.658520000000000216e+00 3.044760000000000133e+00
 H   2.664039999999999964e+00 4.084109999999999907e+00 1.901380000000000070e+00
 O   2.735829999999999984e+00 2.768990000000000062e+00 8.625249999999999861e-01
 H   3.225159999999999805e+00 2.987849999999999895e+00 2.081399999999999917e-02
 H   3.046959999999999891e+00 1.855429999999999913e+00 1.160919999999999952e+00
 O   6.286438999999999666e+00 2.290369999999999795e+00 8.821559999999999846e+00
 H   5.968110000000000248e+00 2.744099999999999984e+00 8.033289999999999154e+00
 H   6.777009999999999756e+00 3.062409999999999854e+00 9.332560000000000855e+00
 O   7.828409999999999869e+00 1.507100000000000106e-01 8.392839999999999634e+00
 H   7.342240000000000322e+00 9.850069999999999659e-01 8.599750000000000227e+00
 H   8.293810000000000571e+00 4.392679999999999918e-01 7.536319999999999908e+00
 O   8.023160000000000736e+00 4.406859999999999999e+00 3.091870000000000118e+00
 H   8.394849000000000672e+00 5.309750000000000192e+00 3.195619999999999905e+00
 H   8.640340999999999383e+00 3.670160000000000089e+00 3.449829999999999952e+00
"""

cell.basis = 'gth-tzv2p'
cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
cell.max_memory = 16000 # in MB
cell.precision = 1e-8 # integral precision
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.use_loose_rcut = True # integral screening based on shell radii
cell.use_particle_mesh_ewald = True # use particle mesh ewald for nuclear repulsion
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
