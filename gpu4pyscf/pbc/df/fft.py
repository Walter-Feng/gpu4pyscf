import pyscf
from pyscf.pbc.df.fft import FFTDF as cpu_FFTDF
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc import tools
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
import numpy as np
import cupy
import math

def get_j_kpts(df_object, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band = None, weights = None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''

    cell = df_object.cell
    mesh = df_object.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1

    density_matrices_at_kpts = cupy.asarray(dm_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points = formatted_density_matrices.shape[:2]

    print(df_object.ao_on_grid.conj())
    # Maybe the ao_on_grid of shape (n_channels, n_ao, n_grid_points) cannot be saved.
    density_in_real_space = cupy.einsum('ikpq, knp, knq->in', formatted_density_matrices, df_object.ao_on_grid.conj(), df_object.ao_on_grid)
    density_in_real_space = density_in_real_space.reshape(-1, *mesh)
    density_in_k_space = cupy.fft.fftn(density_in_real_space, axes=(1,2,3))
    coulomb_weighted_density_in_k_space = cupy.einsum("xyz, ixyz -> ixyz", df_object.coulomb_in_k_space, density_in_k_space)
    coulomb_in_real_space = cupy.fft.ifftn(coulomb_weighted_density_in_k_space, axes=(1,2,3)).reshape(n_channels, -1)

    if hermi == 1 or is_zero(kpts):
        coulomb_in_real_space = coulomb_in_real_space.real
        
    weight = cell.vol / df_object.n_grid_points / n_k_points
    coulomb_in_real_space *= weight

    # needs to modify if kpts_band is specified. Does anyone really use custom kpts_band by the way?
    vj_at_kpts_on_gpu = cupy.einsum('knp, in, knq -> ikpq', df_object.ao_on_grid.conj(), coulomb_in_real_space, df_object.ao_on_grid)

    return _format_jks(vj_at_kpts_on_gpu, dm_kpts, kpts_band, kpts)


def get_k_kpts(df_object, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray
            Density matrix at each k-point
        kpts : (nkpts, 3) ndarray

    Kwargs:
        hermi : int
            Whether K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian

        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        vk : (nkpts, nao, nao) ndarray
        or list of vj and vk if the input dm_kpts is a list of DMs
    '''
    cell = df_object.cell
    mesh = df_object.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension > 1
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    density_matrices_at_kpts = cupy.asarray(dm_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points = formatted_density_matrices.shape[:2]

    weight = 1./n_k_points * (cell.vol / ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if is_zero(kpts_band) and is_zero(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = df_object.grids.coords
    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in df_object._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in df_object._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = df_object.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    logger.debug1(df_object, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                  max_memory, blksize)
    #ao1_dtype = np.result_type(*ao1_kpts)
    #ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = (logger.process_clock(), logger.perf_counter())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
        t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


class FFTDF(cpu_FFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        cpu_FFTDF.__init__(self, cell, kpts)
        self.to_gpu()
        numerical_integrator = self._numint
        self.ao_on_grid = cupy.asarray(numerical_integrator.eval_ao(cell, self.grids.coords, kpts))
        self.coulomb_in_k_space = cupy.asarray(tools.get_coulG(cell, mesh=self.mesh)).reshape(*self.mesh)
        self.n_grid_points = math.prod(self.mesh)
    

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        
        j_on_gpu = get_j_kpts(self, dm, hermi, kpts, kpts_band)

        quit()


