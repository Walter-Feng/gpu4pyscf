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
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import multigrid
from gpu4pyscf.pbc.dft.multi_grid import muligrid_pair
import sys
import cProfile, pstats, io
cell=pbcgto.Cell()

#Molecule
boxlen=12.4138
cell.a=numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
cell.atom="""
O 0 0 0,
H 1 0 0,
H 0 1 0
"""
cell.basis = 'gth-tzv2p'
cell.ke_cutoff = 200  # kinetic energy cutoff in a.u.
cell.max_memory = 8000 # in MB
cell.precision = 1e-6 # integral precision
cell.pseudo = 'gth-pade'
cell.verbose = 5
cell.use_loose_rcut = True # integral screening based on shell radii
cell.use_particle_mesh_ewald = True # use particle mesh ewald for nuclear repulsion
cell.build()

mf=pbcdft.RKS(cell)
#mf.xc = "LDA, VWN"
mf.xc = "PBE,PBE"
mf.max_cycle = 1
mf.init_guess = 'atom' # atom guess is fast
mf.with_df = muligrid_pair.MultiGridFFTDF2(cell)
mf.with_df.ngrids = 4 # number of sets of grid points
mf.kernel()


