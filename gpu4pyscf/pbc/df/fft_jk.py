# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
JK with GPW
'''

__all__ = [
    'get_j_kpts', 'get_k_kpts', 'get_jk', 'get_j', 'get_k',
    'get_j_e1_kpts', 'get_k_e1_kpts'
]

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero, member
from pyscf.pbc.df.df_jk import _format_kpts_band
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import tools

def get_j_kpts(df_object, dm_at_kpts, hermi=1, kpts=np.zeros((1, 3)), kpts_band=None, comm=None):
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

    density_matrices_at_kpts = cp.asarray(dm_at_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points, n_ao = formatted_density_matrices.shape[:3]

    data_type = df_object.ao_on_grid[-1].dtype

    # Maybe the ao_on_grid of shape (n_channels, n_ao, n_grid_points) cannot be saved.
    density_in_real_space = cp.zeros((n_channels, df_object.n_grid_points), dtype=data_type)

    for i in range(n_channels):
        for k in range(n_k_points):
            density_in_real_space[i] += cp.einsum('pq, np, nq -> n', formatted_density_matrices[i, k],
                                                    df_object.ao_on_grid[k].conj(),
                                                    df_object.ao_on_grid[k], optimize=True)

    density_in_real_space = density_in_real_space.reshape(-1, *mesh)
    density_in_g_space = cp.fft.fftn(density_in_real_space, axes=(1, 2, 3))
    coulomb = cp.asarray(tools.get_coulG(cell, mesh=mesh)).reshape(*mesh)
    coulomb_weighted_density_in_real_space = cp.fft.ifftn(coulomb * density_in_g_space, axes=(1, 2, 3)).reshape(
        n_channels, -1)

    if is_zero(kpts):
        coulomb_weighted_density_in_real_space = coulomb_weighted_density_in_real_space.real

    weight = cell.vol / df_object.n_grid_points / n_k_points
    coulomb_weighted_density_in_real_space *= weight

    # needs to modify if kpts_band is specified. Does anyone really use custom kpts_band by the way?
    vj_at_kpts_on_gpu = cp.zeros((n_channels, n_k_points, n_ao, n_ao), dtype=data_type)
    for i in range(n_channels):
        for k in range(n_k_points):
            vj_at_kpts_on_gpu[i, k] = cp.einsum('np, n, nq -> pq', df_object.ao_on_grid[k].conj(),
                                                  coulomb_weighted_density_in_real_space[i],
                                                  df_object.ao_on_grid[k], optimize=True)

    return _format_jks(vj_at_kpts_on_gpu, dm_at_kpts, kpts_band, kpts)

def get_k_kpts(df_object, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), kpts_band=None, exxdiv=None, overlap=None):
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
    coords = cp.asarray(df_object.grids.coords)
    if overlap is None:
        overlap = cp.asarray(cell.pbc_inyot("int1e_ovlp", hermi=hermi, kpts=kpts))

    density_matrices_at_kpts = cp.asarray(dm_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points, n_ao = formatted_density_matrices.shape[:3]
    data_type = df_object.ao_on_grid[-1].dtype

    weight = 1. / n_k_points * (cell.vol / df_object.n_grid_points)

    vk = cp.zeros((n_channels, n_k_points, n_ao, n_ao), dtype=data_type)

    for k2_index, k2, ao_at_k2 in zip(range(n_k_points), kpts, df_object.ao_on_grid):
        density_dot_ao_at_k2 = cp.ndarray((n_channels, n_ao, df_object.n_grid_points), dtype=data_type)
        for i in range(n_channels):
            density_dot_ao_at_k2[i] = formatted_density_matrices[i, k2_index] @ ao_at_k2.conj().T
        for k1_index, k1, ao_at_k1 in zip(range(n_k_points), kpts, df_object.ao_on_grid):
            if is_zero(kpts):
                e_ikr = cp.ones(len(coords))
            else:
                e_ikr = cp.exp(coords @ cp.asarray(1j * (k1 - k2)))

            block_size = 32
            for p0, p1 in lib.prange(0, n_ao, block_size):

                ao_sandwiched_e_ikr_in_real_space = cp.einsum(
                    "np, nq, n -> pqn", ao_at_k1[:, p0:p1].conj(), ao_at_k2, e_ikr).reshape(p1 - p0, n_ao, *mesh)

                ao_sandwiched_e_ikr_in_g_space = cp.fft.fftn(ao_sandwiched_e_ikr_in_real_space,
                                                               axes=(2, 3, 4))

                coulomb_in_g_space = cp.asarray(
                    tools.get_coulG(df_object.cell, k=k2 - k1, mf=df_object, mesh=mesh).reshape(*mesh))

                coulomb_in_real_space = cp.fft.ifftn(coulomb_in_g_space * ao_sandwiched_e_ikr_in_g_space,
                                                       axes=(2, 3, 4)).reshape(p1 - p0, n_ao, -1)

                if is_zero(kpts):
                    coulomb_in_real_space = coulomb_in_real_space.real

                for i in range(n_channels):
                    coulomb_weighted_density_dot_ao_at_k2 = cp.einsum("pqn, qn -> pn", coulomb_in_real_space,
                                                                        density_dot_ao_at_k2[i])

                    coulomb_weighted_density_dot_ao_at_k2 *= e_ikr.conj()
                    vk[i, k1_index, p0:p1, :] += weight * coulomb_weighted_density_dot_ao_at_k2 @ ao_at_k1

    if exxdiv == 'ewald':
        for i in range(n_channels):
            vk[i] += df_object.madelung * cp.einsum("kpq, kqr, krs -> kps",
                                                      overlap, formatted_density_matrices[i], overlap)

    return _format_jks(vk, dm_kpts, None, kpts)

def get_jk(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None,
           with_j=True, with_k=True, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpts_band)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpts_band, exxdiv)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None):
    '''Get the Coulomb (J) AO matrix for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band)
    if kpts_band is None:
        vj = vj[:,0,:,:]
    if dm.ndim == 2:
        vj = vj[0]
    return vj


def get_k(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None, exxdiv=None):
    '''Get the Coulomb (J) and exchange (K) AO matrices for the given density matrix.

    Args:
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        kpt : (3,) ndarray
            The "inner" dummy k-point at which the DM was evaluated (or
            sampled).
        kpts_band : (3,) ndarray or (*,3) ndarray
            The "outer" primary k-point at which J and K are evaluated.

    Returns:
        The function returns one J and one K matrix, corresponding to the input
        density matrix (both order and shape).
    '''
    dm = cp.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band, exxdiv)
    if kpts_band is None:
        vk = vk[:,0,:,:]
    if dm.ndim == 2:
        vk = vk[0]
    return vk

get_j_e1_kpts = NotImplemented
get_k_e1_kpts = NotImplemented

def _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band=None):
    from pyscf.pbc.tools.pbc import madelung
    s = cp.asarray(cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
    m = madelung(cell, kpts)
    if kpts is None:
        for i,dm in enumerate(dms):
            vk[i] += m * s.dot(dm).dot(s)
    elif np.shape(kpts) == (3,):
        if kpts_band is None or is_zero(kpts_band-kpts):
            for i,dm in enumerate(dms):
                vk[i] += m * s.dot(dm).dot(s)

    elif kpts_band is None or np.array_equal(kpts, kpts_band):
        for k in range(len(kpts)):
            for i,dm in enumerate(dms):
                vk[i,k] += m * s[k].dot(dm[k]).dot(s[k])
    else:
        for k, kpt in enumerate(kpts):
            for kp in member(kpt, kpts_band.reshape(-1,3)):
                for i,dm in enumerate(dms):
                    vk[i,kp] += m * s[k].dot(dm[k]).dot(s[k])
    return vk

def _format_dms(dm_kpts, kpts):
    nkpts = len(kpts)
    nao = dm_kpts.shape[-1]
    dms = dm_kpts.reshape(-1,nkpts,nao,nao)
    assert dms.dtype in (np.double, np.complex128)
    return cp.asarray(dms, order='C')

def _format_jks(v_kpts, dm_kpts, kpts_band, kpts):
    if kpts_band is kpts or kpts_band is None:
        return v_kpts.reshape(dm_kpts.shape)
    else:
        assert v_kpts.ndim == 4 # (Ndm,Nk,Nao,Nao)
        # dm_kpts.shape     kpts.shape     nset
        # (Nao,Nao)         (1 ,3)         None
        # (Ndm,Nao,Nao)     (1 ,3)         Ndm
        # (Nk,Nao,Nao)      (Nk,3)         None
        # (Ndm,Nk,Nao,Nao)  (Nk,3)         Ndm
        if kpts_band.ndim == 1:
            assert dm_kpts.ndim <= 3
            v_kpts = v_kpts[:,0]
            if dm_kpts.ndim < 3: # RHF dm
                v_kpts = v_kpts[0]
        else:
            assert kpts.ndim == 2
            assert dm_kpts.ndim >= 3
            if dm_kpts.ndim == 3: # KRHF dms
                assert len(dm_kpts) == len(kpts)
                v_kpts = v_kpts[0]
            else:  # KUHF dms
                assert v_kpts.shape[1] == len(kpts_band)
        return v_kpts
