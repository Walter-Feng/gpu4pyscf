import pyscf.pbc.scf.khf as cpu_KHF
from pyscf.scf import hf as mol_hf
import gpu4pyscf
import gpu4pyscf.pbc.df.fft as gpu_fft
import numpy as np
import cupy
from pyscf import lib
import gpu4pyscf.scf.hf as gpu_hf
from gpu4pyscf.scf import diis
from gpu4pyscf.lib import logger


def inverse_square_root(matrix):
    eigenvalues, eigenvectors = cupy.linalg.eigh(matrix)
    return cupy.einsum("kpq, kq, krq -> kpr", eigenvectors, 1.0 / cupy.sqrt(eigenvalues), eigenvectors.conj())


class KSCF(gpu_hf.SCF, cpu_KHF.KSCF):

    def __init__(self, cell, kpts=np.zeros((1, 3)), exxdiv='ewald'):
        self.rsjk = None
        mol_hf.SCF.__init__(self, cell)
        self.cell = cell
        self.with_df = gpu_fft.FFTDF(cell, kpts=kpts)
        self.exxdiv = exxdiv
        self.kpts = kpts
        self.conv_tol = max(cell.precision * 10, 1e-8)
        self.exx_built = False
        self.overlap = cupy.asarray(cpu_KHF.KSCF.get_ovlp(self, cell, kpts))
        self.inv_sqrt_overlap = inverse_square_root(self.overlap)
        self.sqrt_overlap = cupy.linalg.inv(self.inv_sqrt_overlap)
        self.hcore = cupy.asarray(cpu_KHF.KSCF.get_hcore(self, cell, kpts))

    def make_rdm1(self, mo_coeff_kpts=None, mo_occ_kpts=None):
        if mo_coeff_kpts is None:
            # Note: this is actually "self.mo_coeff_kpts"
            # which is stored in self.mo_coeff of the scf.hf.RHF superclass
            mo_coeff_kpts = self.mo_coeff
        if mo_occ_kpts is None:
            # Note: this is actually "self.mo_occ_kpts"
            # which is stored in self.mo_occ of the scf.hf.RHF superclass
            mo_occ_kpts = self.mo_occ

        return cupy.einsum("kpq, kq, krq -> kpr", mo_coeff_kpts, mo_occ_kpts, mo_coeff_kpts.conj())

    def eig(self, fock_at_k_points, overlap_at_k_points):
        transformed_fock = cupy.einsum("kpq, kqr, krs -> kps", self.inv_sqrt_overlap, fock_at_k_points,
                                       self.inv_sqrt_overlap)

        eigenvalues, eigenvectors = cupy.linalg.eigh(transformed_fock)

        return eigenvalues, cupy.einsum("kpq, kqr -> kpr", self.sqrt_overlap, eigenvectors)

    def get_hcore(self, cell=None, kpts=None):
        return self.hcore

    def get_ovlp(self, cell=None, kpts=None):
        return self.overlap

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.rsjk:
            vj, vk = self.rsjk.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                      with_j, with_k, omega=omega, exxdiv=self.exxdiv)
        else:
            vj, vk = self.with_df.get_jk(dm_kpts, hermi, kpts, kpts_band,
                                         with_j, with_k, omega=omega, exxdiv=self.exxdiv, overlap=self.get_ovlp())
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        if dm_kpts is None:
            dm_kpts = self.make_rdm1()
        vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        return vj - vk * .5

    def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        '''Following pyscf.scf.hf.energy_elec()
        '''
        if dm_kpts is None: dm_kpts = mf.make_rdm1()
        if h1e_kpts is None: h1e_kpts = mf.get_hcore()
        if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

        nkpts = len(dm_kpts)
        e1 = 1. / nkpts * cupy.einsum('kij,kji', dm_kpts, h1e_kpts)
        e_coul = 1. / nkpts * cupy.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
        mf.scf_summary['e1'] = e1.real
        mf.scf_summary['e2'] = e_coul.real
        logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
        if cpu_KHF.CHECK_COULOMB_IMAG and abs(e_coul.imag > mf.cell.precision * 10):
            logger.warn(mf, "Coulomb energy has imaginary part %s. "
                            "Coulomb integrals (e-e, e-N) may not converge !",
                        e_coul.imag)
        return (e1 + e_coul).real, e_coul.real

    def get_occ(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        if mo_energy_kpts is None: mo_energy_kpts = mf.mo_energy

        nkpts = len(mo_energy_kpts)
        nocc = mf.cell.tot_electrons(nkpts) // 2

        mo_energy = cupy.sort(cupy.hstack(mo_energy_kpts))
        fermi = mo_energy[nocc - 1]
        mo_occ_kpts = []
        for mo_e in mo_energy_kpts:
            mo_occ_kpts.append((mo_e <= fermi).astype(cupy.double) * 2)

        if nocc < mo_energy.size:
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc - 1], mo_energy[nocc])
            if mo_energy[nocc - 1] + 1e-3 > mo_energy[nocc]:
                logger.warn(mf, 'HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc - 1], mo_energy[nocc])
        else:
            logger.info(mf, 'HOMO = %.12g', mo_energy[nocc - 1])

        if mf.verbose >= logger.DEBUG:
            cupy.set_printoptions(threshold=len(mo_energy))
            logger.debug(mf, '     k-point                  mo_energy')
            for k, kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                             k, kpt[0], kpt[1], kpt[2],
                             cupy.sort(mo_energy_kpts[k][mo_occ_kpts[k] > 0]),
                             cupy.sort(mo_energy_kpts[k][mo_occ_kpts[k] == 0]))
            cupy.set_printoptions(threshold=1000)

        return cupy.asarray(mo_occ_kpts)

    def check_sanity(self):
        lib.StreamObject.check_sanity(self)
        if (isinstance(self.exxdiv, str) and self.exxdiv.lower() != 'ewald' and
                isinstance(self.with_df, gpu_fft.FFTDF)):
            logger.warn(self, 'exxdiv %s is not supported in DF or MDF',
                        self.exxdiv)

        if self.verbose >= logger.DEBUG:
            s = self.get_ovlp()
            cond = np.max(lib.cond(s.get()))
            if cond * 1e-17 > self.conv_tol:
                logger.warn(self, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                                  'SCF may be inaccurate and hard to converge.', cond)
        return self

    def get_grad(self, mo_coeff_kpts, mo_occ_kpts, fock):
        '''
        returns 1D array of gradients, like non K-pt version
        note that occ and virt indices of different k pts now occur
        in sequential patches of the 1D array
        '''
        nkpts = len(mo_occ_kpts)
        grad_kpts = [gpu_hf.get_grad(mo_coeff_kpts[k], mo_occ_kpts[k], fock[k])
                     for k in range(nkpts)]
        return cupy.hstack(grad_kpts)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                 fock_last=None):
        h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
        if h1e_kpts is None: h1e_kpts = self.get_hcore()
        if vhf_kpts is None: vhf_kpts = self.get_veff(self.cell, dm_kpts)
        f_kpts = h1e_kpts + vhf_kpts
        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f_kpts

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        if s_kpts is None: s_kpts = self.get_ovlp()
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        if 0 <= cycle < diis_start_cycle - 1 and abs(damp_factor) > 1e-4:
            f_kpts = [gpu_hf.damping(S, D * .5, F, damp_factor) for (S, D, F) in zip(s_kpts, dm_kpts, f_kpts)]
        if diis and cycle >= diis_start_cycle:
            f_kpts = [diis.update(S, D, F, self, h1e, vhf) for (S, D, F) in zip(s_kpts, dm_kpts, f_kpts)]
        if abs(level_shift_factor) > 1e-4:
            f_kpts = [gpu_hf.level_shift(S, D * .5, F, level_shift_factor) for (S, D, F) in
                      zip(s_kpts, dm_kpts, f_kpts)]
        return cupy.asarray(f_kpts)

    kernel = gpu_hf._kernel
    scf = gpu_hf.scf
    DIIS = diis.SCF_DIIS
    diis = DIIS
    diis_space = gpu_hf.SCF.diis_space
    diis_damp = gpu_hf.SCF.diis_damp
    diis_start_cycle = gpu_hf.SCF.diis_start_cycle
    diis_file = gpu_hf.SCF.diis_file
    diis_space_rollback = gpu_hf.SCF.diis_space_rollback


class KRHF(KSCF):
    def get_init_guess(self, cell=None, key='minao', s1e=None):
        if s1e is None:
            s1e = self.get_ovlp(cell)
        dm = cupy.asarray(gpu_hf.SCF.get_init_guess(self, cell, key))
        nkpts = len(self.kpts)
        if dm.ndim == 2:
            # dm[nao,nao] at gamma point -> dm_kpts[nkpts,nao,nao]
            dm = cupy.repeat(dm[None, :, :], nkpts, axis=0)

        ne = cupy.einsum('kij,kji->', dm, s1e).real
        # FIXME: consider the fractional num_electron or not? This maybe
        # relate to the charged system.
        nelectron = float(self.cell.tot_electrons(nkpts))
        if abs(ne - nelectron) > 0.01 * nkpts:
            logger.debug(self, 'Big error detected in the electron number '
                               'of initial guess density matrix (Ne/cell = %g)!\n'
                               '  This can cause huge error in Fock matrix and '
                               'lead to instability in SCF for low-dimensional '
                               'systems.\n  DM is normalized wrt the number '
                               'of electrons %s', ne / nkpts, nelectron / nkpts)
            dm *= (nelectron / ne).reshape(-1, 1, 1)
        return dm
