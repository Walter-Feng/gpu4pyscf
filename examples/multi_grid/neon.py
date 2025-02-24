import cupy.cuda
from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from gpu4pyscf.pbc import dft as pbcdft
from pyscf.pbc import dft as cpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid
from pyscf.pbc.dft import multigrid as cpu_multi_grid


diamond_cell = bulk('Ne', 'sc', a=6)

with cupy.cuda.Device(0):
    lattice_vectors = diamond_cell.cell
    cell = gto.M(
        h=np.array(lattice_vectors),
        atom=ase_atoms_to_pyscf(bulk('Ne', 'sc', a=6)),
        basis='minao',
        verbose=6,
        unit='aa',
        ke_cutoff=200
    )
    cell.exp_to_discard = 0.1

    mf = pbcdft.RKS(cell)
    # mf.xc = "LDA, VWN"
    mf.xc = "LDA"
    #mf.max_cycle = 0
    mf = multi_grid.fftdf(mf)
    mf.with_df.ngrids = 4  # number of sets of grid points
    mf.kernel()

    mf = cpu_pbcdft.RKS(cell)
    # mf.xc = "LDA, VWN"
    mf.xc = "LDA"
    #mf.max_cycle = 0
    mf = cpu_multi_grid.multigrid.multigrid_fftdf(mf)
    mf.with_df.ngrids = 4  # number of sets of grid points
    mf.kernel()