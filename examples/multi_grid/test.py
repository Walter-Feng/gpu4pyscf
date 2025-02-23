#from os.path import expanduser
#home_dir = expanduser("~")
#f = open(home_dir+'/.pyscf_conf.py', 'a')
# use FFTW for fft, this requires to compile the FFTW library
# cmake -DENABLE_FFTW=ON -DBUILD_FFTW=ON
#f.write('pbc_tools_pbc_fft_engine=\'FFTW\'')
#f.close()

import numpy
import pyscf
from pyscf import lib
from pyscf import pbc
from pyscf.pbc import gto
from pyscf.pbc import dft
from pyscf.pbc.dft import multigrid as cpu_multi_grid

import cupy.cuda
from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid

import sys
import cProfile, pstats, io
cell=gto.Cell()



#Molecule
boxlen=6
cell.a=numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom="""
N 1 1 1,
H 1 1 0,
H 1 0 1,
H 0 1 1
"""
cell.basis = 'minao'
cell.ke_cutoff = 50  # kinetic energy cutoff in a.u.
cell.max_memory = 8000 # in MB
cell.precision = 1e-20# integral precision
cell.pseudo = 'gth-pade'
cell.verbose = 5
# cell.use_loose_rcut = True # integral screening based on shell radii
# .use_particle_mesh_ewald = True # use particle mesh ewald for nuclear repulsion
cell.build()

gpu_mf = gpu_pbcdft.RKS(cell)
gpu_mf.xc = "LDA"
gpu_mf.max_cycle = 10
gpu_mf = multi_grid.fftdf(gpu_mf)
gpu_mf.with_df.ngrids = 4
gpu_mf.kernel()

# mf=dft.RKS(cell)
# mf.xc = "LDA"
# # mf.xc = "PBE,PBE"
# mf.max_cycle = 10
# mf.init_guess = 'atom' # atom guess is fast
# mf.with_df = cpu_multi_grid.MultiGridFFTDF2(cell)
# mf.with_df.ngrids = 4 # number of sets of grid points
# mf.kernel()

