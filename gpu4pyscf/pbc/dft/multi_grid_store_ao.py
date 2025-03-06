import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.lib.cupy_helper as cupy_helper

from pyscf.pbc.dft.multigrid import multigrid as cpu_multi_grid
import pyscf.pbc.gto as gto
import pyscf.gto as mol_gto
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib as cpu_lib
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int

import numpy as np
import cupy as cp

libgpbc = cupy_helper.load_library('libgpbc')

PTR_COORD = 1
EIJ_CUTOFF = 60


def generate_ao_values(mydf, kpts):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    numerical_integrator = mydf._numint

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        tasks = cpu_multi_grid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    ao_values = []

    for grids_dense, grids_sparse in tasks:
        ao_values_for_this_task = {}
        ao_for_dense = numerical_integrator.eval_ao(grids_dense.cell, grids_dense.coords, kpts)
        ao_values_for_this_task["dense"] = ao_for_dense
        if grids_sparse is not None:
            ao_values_for_this_task["sparse"] = numerical_integrator.eval_ao(grids_sparse.cell,  grids_dense.coords, kpts)
        else:
            ao_values_for_this_task["sparse"] = None
        ao_values.append(ao_values_for_this_task)
        

    mydf.ao_values = ao_values


def evaluate_density_on_g_mesh(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0, rho_g_high_order=None):
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_channels, n_k_points, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        raise "Tasks should not be None. Let me fix later"

    assert (deriv < 1)
    density_slices = 1  # Presumably
    if deriv == 0:
        xc_type = 'LDA'

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros((n_channels * density_slices, nx, ny, nz), dtype=cp.complex128)
    for (grids_dense, grids_sparse), ao_values in zip(tasks, mydf.ao_values):

        mesh = grids_dense.mesh
        fft_grids = list(map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))
        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1. / n_k_points * mydf.cell.vol / n_grid_points

        # The first pass handles all diffused functions using the regular
        # matrix multiplication code.
        density = cp.zeros((n_channels, density_slices, n_grid_points), dtype=cp.complex128)
        density_matrix_in_dense_region = dms[:, :, grids_dense.ao_idx[:, None], grids_dense.ao_idx]
        for k in range(n_k_points):
            for i in range(n_channels):
                if xc_type == 'LDA':
                    ao_dot_dm = cp.dot(ao_values["dense"][k], density_matrix_in_dense_region[i, k])
                    density_subblock = cp.einsum('xi,xi->x', ao_dot_dm, ao_values["dense"][k].conj())
                density[i, :] += density_subblock
            
        if grids_sparse is not None:
            density_matrix_slice = dms[:, :, grids_dense.ao_idx[:, None], grids_sparse.ao_idx]
            for k in range(n_k_points):
                for i in range(n_channels):
                    if xc_type == 'LDA':
                        ao_dot_dm = cp.dot(ao_values["dense"][k], density_matrix_slice[i, k])
                        density_subblock = cp.einsum('xi,xi->x', ao_dot_dm, ao_values["sparse"][k].conj())
                    density[i, :] += 2 * density_subblock
                    
        if hermi:
            density = density.real

        density_contribution_on_g_mesh = tools.fft(density.reshape(n_channels * density_slices, -1),
                                                   mesh) * weight_per_grid_point

        density_on_g_mesh[
            cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)] += density_contribution_on_g_mesh.reshape(
            (-1,) + tuple(mesh))

    density_on_g_mesh = density_on_g_mesh.reshape(n_channels, density_slices, -1)
    return density_on_g_mesh

def convert_xc_on_g_mesh_to_fock(mydf, xc_on_g_mesh, hermi=1, kpts=np.zeros((1, 3)), verbose=None):
    cell = mydf.cell
    n_k_points = len(kpts)
    nao = cell.nao_nr()
    xc_on_g_mesh = xc_on_g_mesh.reshape(-1, *mydf.mesh)
    n_channels = xc_on_g_mesh.shape[0]

    at_gamma_point = cpu_multi_grid.gamma_point(kpts)

    if hermi != 1:
        raise NotImplementedError

    data_type = cp.float64
    if not at_gamma_point:
        data_type = cp.complex128

    fock = cp.zeros((n_channels, n_k_points, nao, nao), dtype=data_type)

    for (grids_dense, grids_sparse), ao_values in zip(mydf.tasks, mydf.ao_values):
        mesh = grids_dense.mesh
        n_grid_points = np.prod(mesh)
        fft_grids = map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh)
        interpolated_xc_on_g_mesh = xc_on_g_mesh[
            cp.ix_(cp.arange(xc_on_g_mesh.shape[0]), *fft_grids)].reshape(n_channels, n_grid_points)

        reordered_xc_on_real_mesh = tools.ifft(interpolated_xc_on_g_mesh, mesh).reshape(n_channels, n_grid_points)
        # order='C' forces a copy. otherwise the array is not contiguous
        reordered_xc_on_real_mesh = cp.asarray(reordered_xc_on_real_mesh.real, order='C')
        for k in range(n_k_points):
            for i in range(n_channels):
                xc_scaled_ao = numint._scale_ao(ao_values["dense"][k], reordered_xc_on_real_mesh[i])
                xc_sub_block = cp.dot(ao_values["dense"][k].conj().T, xc_scaled_ao)
                fock[i, k, grids_dense.ao_idx[:, None], grids_dense.ao_idx] += xc_sub_block

        if grids_sparse is not None:
            for k in range(n_k_points):
                for i in range(n_channels):
                    xc_scaled_ao = numint._scale_ao(ao_values["sparse"][k], reordered_xc_on_real_mesh[i])
                    xc_sub_block = cp.dot(ao_values["dense"][k].conj().T, xc_scaled_ao)
                    fock[i, k, grids_dense.ao_idx[:, None], grids_sparse.ao_idx] += xc_sub_block
                    fock[i, k, grids_sparse.ao_idx[:, None], grids_dense.ao_idx] += xc_sub_block.T.conj()

    return fock


def get_nuc(mydf, kpts=None):
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    charge = cp.asarray(-cell.atom_charges())
    Gv = cell.get_Gv(mesh)
    SI = cp.asarray(cell.get_SI(Gv))
    rhoG = cp.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    hermi = 1
    vne = convert_xc_on_g_mesh_to_fock(mydf, vneG, hermi, kpts)[0]

    if is_single_kpt:
        vne = vne[0]
    return vne

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed.
    '''
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    Gv = cell.get_Gv(mesh)

    ngrids = len(Gv)
    vpplocG = cp.empty((ngrids,), dtype=cp.complex128)

    for ig0, ig1 in cpu_lib.prange(0, ngrids, ngrids):
        vpplocG_batch = cp.asarray(
            pp_int.get_gth_vlocG_part1(cell, Gv[ig0:ig1]))
        SI = cell.get_SI(Gv[ig0:ig1])
        vpplocG[ig0:ig1] = -cp.einsum('ij,ij->j', SI, vpplocG_batch)

    hermi = 1
    vpp = convert_xc_on_g_mesh_to_fock(mydf, vpplocG, hermi, kpts)[0]
    vpp2 = cp.asarray(pp_int.get_pp_loc_part2(cell, kpts))
    for k, kpt in enumerate(kpts):
        vpp[k] += vpp2[k]

    # vppnonloc evaluated in reciprocal space
    fakemol = mol_gto.Mole()
    fakemol._atm = np.zeros((1, mol_gto.ATM_SLOTS), dtype=cp.int32)
    fakemol._bas = np.zeros((1, mol_gto.BAS_SLOTS), dtype=cp.int32)
    ptr = mol_gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr + 10)
    fakemol._bas[0, mol_gto.NPRIM_OF] = 1
    fakemol._bas[0, mol_gto.NCTR_OF] = 1
    fakemol._bas[0, mol_gto.PTR_EXP] = ptr + 3
    fakemol._bas[0, mol_gto.PTR_COEFF] = ptr + 4

    def vppnl_by_k(kpt):
        SPG_lm_aoGs = []
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                SPG_lm_aoGs.append(None)
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p1 = p1 + nl * (l * 2 + 1)
            SPG_lm_aoGs.append(
                np.zeros((p1, cell.nao), dtype=np.complex128))

        vppnl = 0
        for ig0, ig1 in cpu_lib.prange(0, ngrids, ngrids):
            ng = ig1 - ig0
            # buf for SPG_lmi upto l=0..3 and nl=3
            buf = np.empty((48, ng), dtype=np.complex128)
            Gk = Gv[ig0:ig1] + kpt
            G_rad = np.linalg.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv[ig0:ig1], kpt=kpt) * (ngrids / cell.vol)
            for ia in range(cell.natm):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        fakemol._bas[0, mol_gto.ANG_OF] = l
                        fakemol._env[ptr + 3] = .5 * rl ** 2
                        fakemol._env[ptr + 4] = rl ** (
                                l + 1.5) * np.pi ** 1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)

                        p0, p1 = p1, p1 + nl * (l * 2 + 1)
                        # pYlm is real, SI[ia] is complex
                        pYlm = np.ndarray((nl, l * 2 + 1, ng),
                                          dtype=np.complex128,
                                          buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad * rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                        #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                        #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                        #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
                if p1 > 0:
                    SPG_lmi = buf[:p1]
                    SPG_lmi *= cell.get_SI(Gv[ig0:ig1], atmlst=[ia, ]).conj()
                    SPG_lm_aoGs[ia] += cpu_lib.zdot(SPG_lmi, aokG)
            buf = None
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    p0, p1 = p1, p1 + nl * (l * 2 + 1)
                    hl = np.asarray(hl)
                    SPG_lm_aoG = SPG_lm_aoGs[ia][p0:p1].reshape(nl, l * 2 + 1,
                                                                -1)
                    tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        SPG_lm_aoGs = None
        return vppnl * (1. / ngrids ** 2)

    for k, kpt in enumerate(kpts):
        vppnl = cp.asarray(vppnl_by_k(kpt))
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if is_single_kpt:
        vpp = vpp[0]
    return vpp

def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    numerical_integrator = mydf._numint
    xc_type = numerical_integrator._xc_type(xc_code)

    if xc_type == 'LDA':
        derivative_order = 0
    else:
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    
    density_on_G_mesh = evaluate_density_on_g_mesh(mydf, dm_kpts, hermi, kpts, derivative_order)
    t0 = log.timer('density', *t0)
    coulomb_kernel_on_g_mesh = tools.get_coulG(cell, mesh=mesh)
    coulomb_on_g_mesh = cp.einsum('ng,g->ng', density_on_G_mesh[:, 0], coulomb_kernel_on_g_mesh)
    coulomb_energy = .5 * cp.einsum('ng,ng->n', density_on_G_mesh[:, 0].real, coulomb_on_g_mesh.real)
    coulomb_energy += .5 * cp.einsum('ng,ng->n', density_on_G_mesh[:, 0].imag, coulomb_on_g_mesh.imag)
    coulomb_energy /= cell.vol
    log.debug('Multigrid Coulomb energy %s', coulomb_energy)
    t0 = log.timer('coulomb', *t0)
    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density_in_real_space = tools.ifft(density_on_G_mesh.reshape(-1, ngrids), mesh).real * (1. / weight)
    density_in_real_space = density_in_real_space.reshape(nset, -1, ngrids)
    n_electrons = density_in_real_space[:, 0].sum(axis=1) * weight

    weighted_xc_for_fock_on_g_mesh = cp.ndarray((nset, *density_in_real_space.shape), dtype=cp.complex128)
    xc_energy_sum = cp.zeros(nset)
    for i in range(nset):
        if xc_type == 'LDA':
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i, 0], deriv=1,
                                                                          xctype=xc_type)[:2]
        else:
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i], deriv=1,
                                                                          xctype=xc_type)[:2]

        xc_energy_sum[i] += (density_in_real_space[i, 0] * xc_for_energy.flatten()).sum() * weight

        weighted_xc_for_fock_on_g_mesh[i] = tools.fft(xc_for_fock * weight, mesh)
    density_in_real_space = density_on_G_mesh = None

    if nset == 1:
        coulomb_energy = coulomb_energy[0]
        n_electrons = n_electrons[0]
        xc_energy_sum = xc_energy_sum[0]
    log.debug('Multigrid exc %s  nelec %s', xc_energy_sum, n_electrons)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xc_type == 'LDA':
        if with_j:
            weighted_xc_for_fock_on_g_mesh[:, 0] += coulomb_on_g_mesh

        xc_for_fock = convert_xc_on_g_mesh_to_fock(mydf, weighted_xc_for_fock_on_g_mesh, hermi, kpts_band)

    else:
        raise NotImplementedError

    if return_j:
        vj = convert_xc_on_g_mesh_to_fock(mydf, coulomb_on_g_mesh, hermi, kpts_band, verbose=log)
        vj = fft_jk._format_jks(vj, dm_kpts, input_band, kpts)
    else:
        vj = None

    t0 = log.timer('xc', *t0)
    
    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    xc_for_fock = xc_for_fock.reshape(shape)
    xc_for_fock = cupy_helper.tag_array(xc_for_fock, ecoul=coulomb_energy, exc=xc_energy_sum, vj=vj, vk=None)
    return n_electrons, xc_energy_sum, xc_for_fock


class FFTDF(fft.FFTDF, cpu_multi_grid.MultiGridFFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        self.sorted_gaussian_pairs = None
        fft.FFTDF.__init__(self, cell, kpts)
        generate_ao_values(self, kpts)

    get_nuc = get_nuc
    get_pp = get_pp
    
    
def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
