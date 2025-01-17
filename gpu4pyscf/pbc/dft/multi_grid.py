import pyscf.pbc.gto as gto
import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from pyscf.dft.numint import libdft

import gpu4pyscf.pbc.scf.hf
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.lib.cupy_helper as cupy_helper

from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band

import numpy as np
import cupy as cp
import scipy

import ctypes

libgdft = cupy_helper.load_library('libgdft')


def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, mesh=None, offset=None, submesh=None):
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(mesh) / vol
    exp_min = np.hstack(cell.bas_exps()).min()
    theta_ij = exp_min / 2
    lattice_sum_fac = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_sum_fac
    if xctype != 'LDA':
        precision *= .1
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[multigrid.PTR_EXPDROP] = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    Ls = gto.eval_gto.get_lattice_Ls(cell)
    nimgs = len(Ls)

    weights = weights.get()
    assert (weights.dtype == np.double)
    xctype = xctype.upper()
    n_mat = None
    if xctype == 'LDA':
        if weights.ndim == 1:
            weights = weights.reshape(-1, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    a = cell.lattice_vectors()
    b = np.linalg.inv(a.T)
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh
    # log_prec is used to estimate the gto_rcut. Add EXTRA_PREC to count
    # other possible factors and coefficients in the integral.
    log_prec = np.log(precision * multigrid.EXTRA_PREC)

    if abs(a - np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c

    def make_mat(weights):
        mat = np.zeros((nimgs, comp, naoj, naoi))
        drv(getattr(libdft, eval_fn),
            weights.ctypes.data_as(ctypes.c_void_p),
            mat.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int * 4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*submesh),
            (ctypes.c_int * 3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(mat)

    out = []
    for wv in weights:
        if cell.dimension == 0:
            mat = make_mat(wv)[0].transpose(0, 2, 1)
            if hermi == 1:
                for i in range(comp):
                    cupy_helper.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
        elif kpts is None or multigrid.gamma_point(kpts):
            mat = make_mat(wv).sum(axis=0).transpose(0, 2, 1)
            if hermi == 1:
                for i in range(comp):
                    cupy_helper.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
            if getattr(kpts, 'ndim', None) == 2:
                mat = mat[None, :]
        else:
            mat = make_mat(wv)
            expkL = cp.exp(1j * kpts.reshape(-1, 3).dot(Ls.T))
            mat = cp.einsum('kr,rcij->kcij', expkL, mat)
            if hermi == 1:
                for i in range(comp):
                    for k in range(len(kpts)):
                        cp.hermi_triu(mat[k, i], inplace=True)
            mat = mat.transpose(0, 1, 3, 2)
            if comp == 1:
                mat = mat[:, 0]
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out


def eval_rho(cell, dm, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             mesh=None, offset=None, submesh=None, ignore_imag=False,
             out=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Kwargs:
        ignore_image :
            The output density is assumed to be real if ignore_imag=True.
    '''
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(mesh) / vol
    exp_min = np.hstack(cell.bas_exps()).min()
    theta_ij = exp_min / 2
    lattice_sum_fac = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_sum_fac
    if xctype != 'LDA':
        precision *= .1
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[multigrid.PTR_EXPDROP] = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    dm = cp.asarray(dm, order='C')
    assert (dm.shape[-2:] == (naoi, naoj))

    Ls = gto.eval_gto.get_lattice_Ls(cell)

    if cell.dimension == 0 or kpts is None or multigrid.gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(-1, 1, naoi, naoj).transpose(0, 1, 3, 2)
    else:
        expkL = np.exp(1j * kpts.reshape(-1, 3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1, nkpts, naoi, naoj).transpose(0, 1, 3, 2)
    n_dm = dm.shape[0]

    a = cell.lattice_vectors()
    b = np.linalg.inv(a.T)
    if offset is None:
        offset = (0, 0, 0)
    if submesh is None:
        submesh = mesh
    log_prec = np.log(precision * multigrid.EXTRA_PREC)

    if abs(a - np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    xctype = xctype.upper()
    if xctype == 'LDA':
        comp = 1
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        comp = 4
    else:
        raise NotImplementedError('meta-GGA')
    if comp == 1:
        shape = (np.prod(submesh),)
    else:
        shape = (comp, np.prod(submesh))
    eval_fn = 'NUMINTrho_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_rho_drv

    def make_rho_(shape, dm, hermi):
        rho = np.zeros(shape, order='C')
        drv(getattr(libdft, eval_fn),
            rho.ctypes.data_as(ctypes.c_void_p),
            dm.get().ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int * 4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*submesh),
            (ctypes.c_int * 3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(rho)

    rho = []
    for i, dm_i in enumerate(dm):
        if cell.dimension == 0:
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == cp.complex128
            if has_imag:
                dmR = cp.asarray(dm_i.real, order='C')
                dmI = cp.asarray(dm_i.imag, order='C')
            else:
                # make a copy because the dm may be overwritten in the
                # NUMINT_rho_drv inplace
                dmR = cp.array(dm_i, order='C', copy=True)

        elif kpts is None or multigrid.gamma_point(kpts):
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == cp.complex128
            if has_imag:
                dmR = cp.repeat(dm_i.real, nimgs, axis=0)
                dmI = cp.repeat(dm_i.imag, nimgs, axis=0)
            else:
                dmR = cp.repeat(dm_i, nimgs, axis=0)

        else:
            dm_L = cp.dot(expkL.T, dm_i.reshape(nkpts, -1)).reshape(nimgs, naoj, naoi)
            dmR = cp.asarray(dm_L.real, order='C')

            if ignore_imag:
                has_imag = False
            else:
                dmI = cp.asarray(dm_L.imag, order='C')
                has_imag = (hermi == 0 and abs(dmI).max() > 1e-8)
                if (has_imag and xctype == 'LDA' and
                        naoi == naoj and
                        # For hermitian density matrices, the anti-symmetry
                        # character of the imaginary part of the density matrices
                        # can be found by rearranging the repeated images.
                        abs(dm_i - dm_i.conj().transpose(0, 2, 1)).max() < 1e-8):
                    has_imag = False
            dm_L = None

        if has_imag:
            # complex density cannot be updated inplace directly by
            # function NUMINT_rho_drv
            if out is None:
                rho_i = cp.empty(shape, cp.complex128)
                rho_i.real = make_rho_(shape, dmR, 0)
                rho_i.imag = make_rho_(shape, dmI, 0)
            else:
                assert out[i].dtype == cp.complex128
                rho_i = out[i].reshape(shape)
                rho_i.real += make_rho_(shape, dmR, 0)
                rho_i.imag += make_rho_(shape, dmI, 0)
        else:
            if out is None:
                # rho_i needs to be initialized to 0 because rho_i is updated
                # inplace in function NUMINT_rho_drv
                rho_i = make_rho_(shape, dmR, hermi)
            else:
                assert out[i].dtype == cp.double
                rho_i = out[i].reshape(shape)
                rho_i += make_rho_(rho_i.shape, dmR, hermi)
        dmR = dmI = None
        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]
    return rho


def _eval_rho_bra(cell, dms, shls_slice, hermi, xctype, kpts, grids,
                  ignore_imag, log):
    a = cp.asarray(cell.lattice_vectors())
    rmax = a.max()
    mesh = np.asarray(grids.mesh)
    rcut = grids.cell.rcut
    nset = dms.shape[0]
    if xctype == 'LDA':
        rho_slices = 1
    else:
        rho_slices = 4

    if rcut > rmax * multigrid.R_RATIO_SUBLOOP:
        rho = eval_rho(cell, dms, shls_slice, hermi, xctype, kpts,
                       mesh, ignore_imag=ignore_imag)
        return cp.reshape(rho, (nset, rho_slices, np.prod(mesh)))

    if hermi == 1 or ignore_imag:
        rho = cp.zeros((nset, rho_slices) + tuple(mesh))
    else:
        rho = cp.zeros((nset, rho_slices) + tuple(mesh), dtype=cp.complex128)

    b = cp.linalg.inv(a.T)
    ish0, ish1, jsh0, jsh1 = shls_slice
    nshells_j = jsh1 - jsh0
    pcell = cell.copy(deep=False)
    rest_dms = []
    rest_bas = []
    i1 = 0
    for atm_id in set(cell._bas[ish0:ish1, multigrid.ATOM_OF]):
        atm_bas_idx = np.where(cell._bas[ish0:ish1, multigrid.ATOM_OF] == atm_id)[0]
        _bas_i = cell._bas[atm_bas_idx]
        l = _bas_i[:, multigrid.ANG_OF]
        i0, i1 = i1, i1 + sum((l + 1) * (l + 2) // 2)
        sub_dms = dms[:, :, i0:i1]

        atom_position = cp.asarray(cell.atom_coord(atm_id))
        frac_edge0 = b.dot(atom_position - rcut)
        frac_edge1 = b.dot(atom_position + rcut)

        if (np.all(0 < frac_edge0) and np.all(frac_edge1 < 1)):
            pcell._bas = np.vstack((_bas_i, cell._bas[jsh0:jsh1]))
            nshells_i = len(atm_bas_idx)
            sub_slice = (0, nshells_i, nshells_i, nshells_i + nshells_j)

            offset = (frac_edge0 * mesh).astype(int)
            mesh1 = np.ceil(frac_edge1 * mesh).astype(int)
            submesh = mesh1 - offset
            log.debug1('atm %d  rcut %f  offset %s submesh %s',
                       atm_id, rcut, offset, submesh)
            rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                            mesh, offset, submesh, ignore_imag=ignore_imag)
            #:rho[:,:,offset[0]:mesh1[0],offset[1]:mesh1[1],offset[2]:mesh1[2]] += \
            #:        numpy.reshape(rho1, (nset, rhodim) + tuple(submesh))
            gx = np.arange(offset[0], mesh1[0], dtype=np.int32)
            gy = np.arange(offset[1], mesh1[1], dtype=np.int32)
            gz = np.arange(offset[2], mesh1[2], dtype=np.int32)
            multigrid._takebak_5d(rho, np.reshape(rho1, (nset, rho_slices) + tuple(submesh)),
                                  (None, None, gx, gy, gz))
        else:
            log.debug1('atm %d  rcut %f  over 2 images', atm_id, rcut)
            #:rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:                mesh, ignore_imag=ignore_imag)
            #:rho += numpy.reshape(rho1, rho.shape)
            # or
            #:eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
            #:         mesh, ignore_imag=ignore_imag, out=rho)
            rest_bas.append(_bas_i)
            rest_dms.append(sub_dms)
    if rest_bas:
        pcell._bas = np.vstack(rest_bas + [cell._bas[jsh0:jsh1]])
        nshells_i = sum(len(x) for x in rest_bas)
        sub_slice = (0, nshells_i, nshells_i, nshells_i + nshells_j)
        sub_dms = np.concatenate(rest_dms, axis=2)
        # Update rho inplace
        eval_rho(pcell, sub_dms, sub_slice, hermi, xctype, kpts,
                 mesh, ignore_imag=ignore_imag, out=rho)
    return rho.reshape((nset, rho_slices, np.prod(mesh)))


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0, rho_g_high_order=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    assert (deriv < 1)
    # hermi = hermi and abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9
    gga_high_order = False
    if deriv == 0:
        xctype = 'LDA'
        rho_slices = 1  # Presumably

    elif deriv == 1:
        if rho_g_high_order is not None:
            raise NotImplementedError

    elif deriv == 2:  # meta-GGA
        raise NotImplementedError

    ignore_imag = (hermi == 1)

    nx, ny, nz = mydf.mesh
    rhoG = cp.zeros((nset * rho_slices, nx, ny, nz), dtype=cp.complex128)
    for grids_dense, grids_sparse in tasks:
        h_cell = grids_dense.cell
        mesh = tuple(grids_dense.mesh)
        ngrids = np.prod(mesh)
        log.debug('mesh %s  rcut %g', mesh, h_cell.rcut)

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            rho = cp.zeros((nset, rho_slices, ngrids), dtype=cp.complex128)
            idx_h = grids_dense.ao_idx
            dms_hh = cp.asarray(dms[:, :, idx_h[:, None], idx_h], order='C')
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts, deriv):
                ao_h, mask = ao_h_etc[0], ao_h_etc[2]
                for k in range(nkpts):
                    for i in range(nset):
                        if xctype == 'LDA':
                            ao_dm = cp.dot(ao_h[k], dms_hh[i, k])
                            rho_sub = cp.einsum('xi,xi->x', ao_dm, ao_h[k].conj())
                        else:
                            rho_sub = numint.eval_rho(h_cell, ao_h[k], dms_hh[i, k],
                                                      mask, xctype, hermi)
                        rho[i, :, p0:p1] += rho_sub
                ao_h = ao_h_etc = ao_dm = None
            if ignore_imag:
                rho = rho.real
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            idx_t = cp.append(idx_h, idx_l)
            dms_ht = cp.asarray(dms[:, :, idx_h[:, None], idx_t], order='C')
            dms_lh = cp.asarray(dms[:, :, idx_l[:, None], idx_h], order='C')

            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = multigrid._pgto_shells(h_cell)
            nshells_t = multigrid._pgto_shells(t_cell)

            if deriv == 0:
                if hermi == 1:
                    naol, naoh = dms_lh.shape[2:]
                    dms_ht[:, :, :, naoh:] += dms_lh.transpose(0, 1, 3, 2)
                    pgto_dms = cp.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                    shls_slice = (0, nshells_h, 0, nshells_t)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None, ignore_imag=True)
                    rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0,
                                        'LDA', kpts, grids_dense, True, log)
                else:
                    raise NotImplementedError

            elif deriv == 1:
                raise NotImplementedError

        weight = 1. / nkpts * cell.vol / ngrids
        rho_freq = tools.fft(rho.reshape(nset * rho_slices, -1), mesh)
        rho_freq *= weight
        gx = cp.fft.fftfreq(mesh[0], 1. / mesh[0]).astype(cp.int32)
        gy = cp.fft.fftfreq(mesh[1], 1. / mesh[1]).astype(cp.int32)
        gz = cp.fft.fftfreq(mesh[2], 1. / mesh[2]).astype(cp.int32)
        #:rhoG[:,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape((-1,)+mesh)

        reshaped_rho_freq = rho_freq.reshape((-1,) + mesh)

        rhoG[cp.ix_(cp.arange(nset * rho_slices), gx, gy, gz)] += reshaped_rho_freq

    rhoG = rhoG.reshape(nset, rho_slices, -1)

    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        rhoG1 = cp.einsum('np,px->nxp', 1j * rhoG[:, 0], Gv)
        rhoG = cp.concatenate([rhoG, rhoG1], axis=1)
    return rhoG


def _get_j_pass2(mydf, vG, hermi=1, kpts=np.zeros((1, 3)), verbose=None):
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1, nx, ny, nz)
    nset = vG.shape[0]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    at_gamma_point = multigrid.gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = cp.zeros((nset, nkpts, nao, nao))
    else:
        vj_kpts = cp.zeros((nset, nkpts, nao, nao), dtype=cp.complex128)

    for grids_dense, grids_sparse in tasks:
        mesh = grids_dense.mesh
        ngrids = cp.prod(mesh)
        log.debug('mesh %s', mesh)

        gx = np.fft.fftfreq(mesh[0], 1. / mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1. / mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1. / mesh[2]).astype(np.int32)
        #:sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids)
        sub_vG = vG[cp.ix_(cp.arange(vG.shape[0]), gx, gy, gz)].reshape(nset, ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset, ngrids)
        vR = cp.asarray(v_rs.real, order='C')
        vI = cp.asarray(v_rs.imag, order='C')
        ignore_vG_imag = hermi == 1 or abs(vI.sum()) < multigrid.IMAG_TOL
        if ignore_vG_imag:
            v_rs = vR
        elif vj_kpts.dtype == cp.double:
            # ensure result complex array if tddft amplitudes are complex while
            # at gamma point
            vj_kpts = vj_kpts.astype(cp.complex128)

        idx_h = grids_dense.ao_idx
        if grids_sparse is None:
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts):
                ao_h = ao_h_etc[0]
                for k in range(nkpts):
                    for i in range(nset):
                        aow = numint._scale_ao(ao_h[k], v_rs[i, p0:p1])
                        vj_sub = cp.dot(ao_h[k].conj().T, aow)
                        vj_kpts[i, k, idx_h[:, None], idx_h] += vj_sub
                ao_h = ao_h_etc = None
        else:
            idx_h = grids_dense.ao_idx
            idx_l = grids_sparse.ao_idx
            # idx_t = numpy.append(idx_h, idx_l)
            naoh = len(idx_h)

            h_cell = grids_dense.cell
            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = multigrid._pgto_shells(h_cell)
            nshells_t = multigrid._pgto_shells(t_cell)
            shls_slice = (0, nshells_h, 0, nshells_t)
            vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
            # Imaginary part may contribute
            if not ignore_vG_imag:
                vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'LDA', kpts)
                vp = cp.asarray(vp) + cp.asarray(vpI) * 1j
                vpI = None

            vp = cp.einsum('nkpq,pi,qj->nkij', cp.asarray(vp), cp.asarray(h_coeff), cp.asarray(t_coeff))

            vj_kpts[:, :, idx_h[:, None], idx_h] += vp[:, :, :, :naoh]
            vj_kpts[:, :, idx_h[:, None], idx_l] += vp[:, :, :, naoh:]

            if hermi == 1:
                vj_kpts[:, :, idx_l[:, None], idx_h] += \
                    vp[:, :, :, naoh:].transpose(0, 1, 3, 2).conj()
            else:
                shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                vp = eval_mat(t_cell, vR, shls_slice, 1, 0, 'LDA', kpts)
                # Imaginary part may contribute
                if not ignore_vG_imag:
                    vpI = eval_mat(t_cell, vI, shls_slice, 1, 0, 'LDA', kpts)
                    vp = cp.asarray(vp) + cp.asarray(vpI) * 1j
                    vpI = None
                vp = cp.einsum('nkpq,pi,qj->nkij', vp, l_coeff, h_coeff)
                vj_kpts[:, :, idx_l[:, None], idx_h] += vp

    return vj_kpts


def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    ni = mydf._numint
    xctype = ni._xc_type(xc_code)

    if xctype == 'LDA':
        deriv = 0
    else:
        raise NotImplementedError

    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = cp.einsum('ng,g->ng', rhoG[:, 0], coulG)
    ecoul = .5 * cp.einsum('ng,ng->n', rhoG[:, 0].real, vG.real)
    ecoul += .5 * cp.einsum('ng,ng->n', rhoG[:, 0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1, ngrids), mesh).real * (1. / weight)
    rhoR = rhoR.reshape(nset, -1, ngrids)
    nelec = rhoR[:, 0].sum(axis=1) * weight

    wv_freq = []
    excsum = cp.zeros(nset)
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i, 0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]

        excsum[i] += (rhoR[i, 0] * exc.flatten()).sum() * weight
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = cp.asarray(wv_freq).reshape(nset, -1, *mesh)
    rhoR = rhoG = None

    if nset == 1:
        ecoul = ecoul[0]
        nelec = nelec[0]
        excsum = excsum[0]
    log.debug('Multigrid exc %s  nelec %s', excsum, nelec)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xctype == 'LDA':
        if with_j:
            wv_freq[:, 0] += vG.reshape(nset, *mesh)
        veff = _get_j_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    elif xctype == 'GGA':
        raise NotImplementedError

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = fft_jk._format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = cupy_helper.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff


class FFTDF(fft.FFTDF, multigrid.MultiGridFFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        fft.FFTDF.__init__(self, cell, kpts)


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
