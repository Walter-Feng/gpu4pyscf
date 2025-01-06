import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger, utils

from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band

import numpy as np
import cupy as cp


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
        rho_slices = 1 # Presumably

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
            rho = numpy.zeros((nset, rhodim, ngrids), dtype=numpy.complex128)
            idx_h = grids_dense.ao_idx
            dms_hh = numpy.asarray(dms[:, :, idx_h[:, None], idx_h], order='C')
            for ao_h_etc, p0, p1 in mydf.aoR_loop(grids_dense, kpts, deriv):
                ao_h, mask = ao_h_etc[0], ao_h_etc[2]
                for k in range(nkpts):
                    for i in range(nset):
                        if xctype == 'LDA':
                            ao_dm = lib.dot(ao_h[k], dms_hh[i, k])
                            rho_sub = numpy.einsum('xi,xi->x', ao_dm, ao_h[k].conj())
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
            idx_t = numpy.append(idx_h, idx_l)
            dms_ht = numpy.asarray(dms[:, :, idx_h[:, None], idx_t], order='C')
            dms_lh = numpy.asarray(dms[:, :, idx_l[:, None], idx_h], order='C')

            l_cell = grids_sparse.cell
            h_pcell, h_coeff = h_cell.decontract_basis(to_cart=True, aggregate=True)
            l_pcell, l_coeff = l_cell.decontract_basis(to_cart=True, aggregate=True)
            t_cell = h_pcell + l_pcell
            t_coeff = scipy.linalg.block_diag(h_coeff, l_coeff)

            nshells_h = _pgto_shells(h_cell)
            nshells_t = _pgto_shells(t_cell)

            if deriv == 0:
                if hermi == 1:
                    naol, naoh = dms_lh.shape[2:]
                    dms_ht[:, :, :, naoh:] += dms_lh.transpose(0, 1, 3, 2)
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                    shls_slice = (0, nshells_h, 0, nshells_t)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None, ignore_imag=True)
                    rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0,
                                        'LDA', kpts, grids_dense, True, log)
                else:
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                    shls_slice = (0, nshells_h, 0, nshells_t)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None)
                    rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0,
                                        'LDA', kpts, grids_dense, ignore_imag, log)
                    pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_lh, l_coeff, h_coeff)
                    shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                    #:rho += eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:                offset=None, submesh=None)
                    rho += _eval_rho_ket(t_cell, pgto_dms, shls_slice, 0,
                                         'LDA', kpts, grids_dense, ignore_imag, log)

            elif deriv == 1:
                pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_ht, h_coeff, t_coeff)
                shls_slice = (0, nshells_h, 0, nshells_t)
                #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'GGA', kpts,
                #:               ignore_imag=ignore_imag)
                rho = _eval_rho_bra(t_cell, pgto_dms, shls_slice, 0, 'GGA',
                                    kpts, grids_dense, ignore_imag, log)

                pgto_dms = lib.einsum('nkij,pi,qj->nkpq', dms_lh, l_coeff, h_coeff)
                shls_slice = (nshells_h, nshells_t, 0, nshells_h)
                #:rho += eval_rho(t_cell, pgto_dms, shls_slice, 0, 'GGA', kpts,
                #:                ignore_imag=ignore_imag)
                rho += _eval_rho_ket(t_cell, pgto_dms, shls_slice, 0, 'GGA',
                                     kpts, grids_dense, ignore_imag, log)
                if hermi == 1:
                    # \nabla \chi_i DM(i,j) \chi_j was computed above.
                    # *2 for \chi_i DM(i,j) \nabla \chi_j
                    rho[:, 1:4] *= 2
                else:
                    raise NotImplementedError

        weight = 1. / nkpts * cell.vol / ngrids
        rho_freq = tools.fft(rho.reshape(nset * rhodim, -1), mesh)
        rho_freq *= weight
        gx = numpy.fft.fftfreq(mesh[0], 1. / mesh[0]).astype(numpy.int32)
        gy = numpy.fft.fftfreq(mesh[1], 1. / mesh[1]).astype(numpy.int32)
        gz = numpy.fft.fftfreq(mesh[2], 1. / mesh[2]).astype(numpy.int32)
        #:rhoG[:,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape((-1,)+mesh)
        _takebak_4d(rhoG, rho_freq.reshape((-1,) + mesh), (None, gx, gy, gz))

    rhoG = rhoG.reshape(nset, rhodim, -1)

    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        rhoG1 = numpy.einsum('np,px->nxp', 1j * rhoG[:, 0], Gv)
        rhoG = numpy.concatenate([rhoG, rhoG1], axis=1)
    return rhoG


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
    ngrids = numpy.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)
    vG = numpy.einsum('ng,g->ng', rhoG[:, 0], coulG)
    ecoul = .5 * numpy.einsum('ng,ng->n', rhoG[:, 0].real, vG.real)
    ecoul += .5 * numpy.einsum('ng,ng->n', rhoG[:, 0].imag, vG.imag)
    ecoul /= cell.vol
    log.debug('Multigrid Coulomb energy %s', ecoul)

    weight = cell.vol / ngrids
    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    rhoR = tools.ifft(rhoG.reshape(-1, ngrids), mesh).real * (1. / weight)
    rhoR = rhoR.reshape(nset, -1, ngrids)
    nelec = rhoR[:, 0].sum(axis=1) * weight

    wv_freq = []
    excsum = numpy.zeros(nset)
    for i in range(nset):
        if xctype == 'LDA':
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i, 0], deriv=1, xctype=xctype)[:2]
        else:
            exc, vxc = ni.eval_xc_eff(xc_code, rhoR[i], deriv=1, xctype=xctype)[:2]
        excsum[i] += (rhoR[i, 0] * exc).sum() * weight
        wv = weight * vxc
        wv_freq.append(tools.fft(wv, mesh))
    wv_freq = numpy.asarray(wv_freq).reshape(nset, -1, *mesh)
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
        if with_j:
            wv_freq[:, 0] += vG.reshape(nset, *mesh)
        # *.5 because v+v.T is always called in _get_gga_pass2
        wv_freq[:, 0] *= .5
        veff = _get_gga_pass2(mydf, wv_freq, hermi, kpts_band, verbose=log)
    veff = _format_jks(veff, dm_kpts, input_band, kpts)

    if return_j:
        vj = _get_j_pass2(mydf, vG, hermi, kpts_band, verbose=log)
        vj = _format_jks(veff, dm_kpts, input_band, kpts)
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    veff = veff.reshape(shape)
    veff = lib.tag_array(veff, ecoul=ecoul, exc=excsum, vj=vj, vk=None)
    return nelec, excsum, veff


class MultiGridFFTDF(fft.FFTDF, multigrid.MultiGridFFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        fft.FFTDF.__init__(self, cell, kpts)
        multigrid.MultiGridFFTDF.__init__(self, cell, kpts)
