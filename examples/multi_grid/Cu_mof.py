from ase import io
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
from pyscf.pbc import gto
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid
from gpu4pyscf.pbc import dft as gpu_pbcdft

ase_atom = io.read("Cu_mof_dehyd.cif")

print(len(ase_atom))

cell = gto.M(
    atom=ase_atoms_to_pyscf(ase_atom),
    basis="gthdzvpmoloptsr",
    pseudo='gth-pbe',
    verbose=5,
    a=ase_atom.cell,
    ke_cutoff=150,
    precision=1e-6,
)
cell.exp_to_discard = 0.1
cell.build()

mf = gpu_pbcdft.RKS(cell)
mf.xc = "PBE"
mf.with_df.ngrids = 4
mf.kernel()
