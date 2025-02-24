from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from gpu4pyscf.pbc import dft as pbcdft
from gpu4pyscf.pbc.dft import multi_grid


diamond_cell = bulk('S', 'sc', a=4)

lattice_vectors = diamond_cell.cell
cell = gto.M(
    h=np.array(lattice_vectors),
    atom=ase_atoms_to_pyscf(bulk('S', 'sc', a=4)),
    basis='minao',
    verbose=6,
    unit='aa',
    ke_cutoff=200
)
cell.exp_to_discard = 0.1

# cell = tools.super_cell(cell, [2, 2, 2])

mf = pbcdft.RKS(cell)
mf.xc = "LDA"
mf.max_cycle = 0
mf = multi_grid.fftdf(mf)
mf.with_df.ngrids = 4  # number of sets of grid points
mf.kernel()
