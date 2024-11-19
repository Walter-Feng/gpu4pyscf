from gpu4pyscf.scf import hf
from pyscf.pbc.dft.multigrid import multigrid_pair
import numpy as np

class MultiGridFFTDF2(multigrid_pair.MultiGridFFTDF2, hf.SCF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):

        multigrid_pair.MultiGridFFTDF2.__init__(self, cell, kpts)
        hf.SCF.__init__(self, cell)


