import pyscf.pbc.scf.khf as cpu_KHF
from pyscf.lib import logger
from pyscf.scf import hf as mol_hf
import numpy as np


class KSCF(cpu_KHF.KSCF):
    pass


class KRHF(KSCF, cpu_KHF.KRHF):
    pass
