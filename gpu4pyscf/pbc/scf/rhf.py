import sys
import pyscf.pbc.scf.khf as cpu_KHF
from pyscf.scf import hf as mol_hf
import gpu4pyscf.pbc.df.fft as gpu_fft
import numpy as np
import cupy
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array


def inverse_square_root(matrix):
    eigenvalues, eigenvectors = cupy.linalg.eigh(matrix)
    return cupy.einsum("kpq, kq, krq -> kpr", eigenvectors, 1.0 / cupy.sqrt(eigenvalues), eigenvectors.conj())


def _kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
            dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    conv_tol = mf.conv_tol
    cell = mf.cell
    verbose = mf.verbose
    log = logger.new_logger(cell, verbose)
    t0 = log.init_timer()
    if (conv_tol_grad is None):
        conv_tol_grad = conv_tol ** .5
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    if (dm0 is None):
        dm0 = mf.get_init_guess(cell, mf.init_guess)

    dm = cupy.asarray(dm0, order='C')
    if hasattr(dm0, 'mo_coeff') and hasattr(dm0, 'mo_occ'):
        if dm0.ndim == 2:
            mo_coeff = cupy.asarray(dm0.mo_coeff)
            mo_occ = cupy.asarray(dm0.mo_occ)
            occ_coeff = cupy.asarray(mo_coeff[:, mo_occ > 0])
            dm = tag_array(dm, occ_coeff=occ_coeff, mo_occ=mo_occ, mo_coeff=mo_coeff)

    h1e = cupy.asarray(mf.get_hcore(cell))
    s1e = cupy.asarray(mf.get_ovlp(cell))

    vhf = mf.get_veff(cell, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    logger.info(mf, 'init E= %.15g', e_tot)
    t1 = log.timer_debug1('total prep', *t0)
    scf_conv = False

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    for cycle in range(mf.max_cycle):
        t0 = log.init_timer()
        dm_last = dm
        last_hf_e = e_tot

        f = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        t1 = log.timer_debug1('DIIS', *t0)
        mo_energy, mo_coeff = mf.eig(f, s1e)
        t1 = log.timer_debug1('eig', *t1)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        t1 = log.timer_debug1('dm', *t1)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        t1 = log.timer_debug1('veff', *t1)
        e_tot = mf.energy_tot(dm, h1e, vhf)
        t1 = log.timer_debug1('energy', *t1)

        norm_ddm = cupy.linalg.norm(dm - dm_last)
        t1 = log.timer_debug1('total', *t0)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |ddm|= %4.3g',
                    cycle + 1, e_tot, e_tot - last_hf_e, norm_ddm)

        if dump_chk:
            local_variables = locals()
            for key in local_variables:
                value = local_variables[key]
                if (type(value) is cupy.ndarray):
                    local_variables[key] = cupy.asnumpy(value)
            mf.dump_chk(local_variables)

        e_diff = abs(e_tot - last_hf_e)
        norm_gorb = cupy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, f))
        if (e_diff < conv_tol and norm_gorb < conv_tol_grad):
            scf_conv = True
            break

    if (cycle == mf.max_cycle):
        logger.warn("SCF failed to converge")

    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class KSCF(cpu_KHF.KSCF):

    def __init__(self, cell, kpts=np.zeros((1, 3)), exxdiv=None):
        self.rsjk = None
        mol_hf.SCF.__init__(self, cell)
        self.with_df = gpu_fft.FFTDF(cell, kpts=kpts)
        self.exxdiv = exxdiv
        self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)
        self.exx_built = False
        self.hcore = None
        self.overlap = None
        self.inv_sqrt_overlap = None
        self.sqrt_overlap = None

    def make_rdm1(self, mo_coeff_kpts = None, mo_occ_kpts = None):
        return cupy.einsum("kpq, kq, kqr -> kpr", mo_coeff_kpts, mo_occ_kpts, mo_coeff_kpts.T.conj())

    def eig(self, fock_at_k_points, overlap_at_k_points):
        n_k_points = len(fock_at_k_points)
        if self.inv_sqrt_overlap is None:
            self.inv_sqrt_overlap = inverse_square_root(overlap_at_k_points)

        if self.sqrt_overlap is None:
            self.sqrt_overlap = cupy.linalg.inv(self.inv_sqrt_overlap)

        transformed_fock = cupy.einsum("kpq, kqr, krs -> kps", self.inv_sqrt_overlap, fock_at_k_points,
                                       self.inv_sqrt_overlap)

        eigenvalues, eigenvectors = cupy.linalg.eigh(transformed_fock)

        return eigenvalues, cupy.einsum("kpq, kqr -> kpr", self.sqrt_overlap, eigenvectors)

    def get_hcore(self, cell=None, kpts=None):
        if self.hcore is None:
            self.hcore = cupy.asarray(cpu_KHF.KSCF.get_hcore(self, cell, kpts))

        return self.hcore

    def get_ovlp(self, cell=None, kpts=None):
        if self.overlap is None:
            self.overlap = cupy.asarray(cpu_KHF.KSCF.get_ovlp(self, cell, kpts))

        return self.overlap


    def energy_elec(mf, dm=None, h1e=None, vhf=None):
        if dm is None: dm = mf.make_rdm1()
        if h1e is None: h1e = mf.get_hcore()
        if vhf is None: vhf = mf.get_veff(mf.mol, dm)
        e1 = numpy.einsum('ij,ji->', h1e, dm).real
        e_coul = numpy.einsum('ij,ji->', vhf, dm).real * .5
        mf.scf_summary['e1'] = e1
        mf.scf_summary['e2'] = e_coul
        logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
        return e1 + e_coul, e_coul

class KRHF(KSCF):
    pass
