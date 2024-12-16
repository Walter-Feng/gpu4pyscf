import pyscf
from pyscf.pbc.df.fft import FFTDF as cpu_FFTDF
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc import tools
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
import numpy as np
import cupy
import math


def get_j_kpts(df_object, dm_at_kpts, hermi=1, kpts=np.zeros((1, 3)), kpts_band=None, weights=None):
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

    density_matrices_at_kpts = cupy.asarray(dm_at_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points, n_ao = formatted_density_matrices.shape[:3]

    data_type = df_object.ao_on_grid[-1].dtype

    # Maybe the ao_on_grid of shape (n_channels, n_ao, n_grid_points) cannot be saved.
    density_in_real_space = cupy.zeros((n_channels, df_object.n_grid_points), dtype=data_type)

    for i in range(n_channels):
        for k in range(n_k_points):
            density_in_real_space[i] += cupy.einsum('pq, np, nq -> n', formatted_density_matrices[i, k],
                                                    df_object.ao_on_grid[k].conj(),
                                                    df_object.ao_on_grid[k], optimize=True)

    density_in_real_space = density_in_real_space.reshape(-1, *mesh)
    density_in_g_space = cupy.fft.fftn(density_in_real_space, axes=(1, 2, 3))
    coulomb = cupy.asarray(tools.get_coulG(cell, mesh=mesh)).reshape(*mesh)
    coulomb_weighted_density = cupy.einsum("xyz, ixyz -> ixyz", coulomb, density_in_g_space)
    coulomb_weighted_density_in_real_space = cupy.fft.ifftn(coulomb_weighted_density, axes=(1, 2, 3)).reshape(
        n_channels, -1)

    weight = cell.vol / df_object.n_grid_points / n_k_points
    coulomb_weighted_density_in_real_space *= weight

    # needs to modify if kpts_band is specified. Does anyone really use custom kpts_band by the way?
    vj_at_kpts_on_gpu = cupy.zeros((n_channels, n_k_points, n_ao, n_ao), dtype=data_type)
    for i in range(n_channels):
        for k in range(n_k_points):
            vj_at_kpts_on_gpu[i, k] = cupy.einsum('np, n, nq -> pq', df_object.ao_on_grid[k].conj(),
                                                  coulomb_weighted_density_in_real_space[i],
                                                  df_object.ao_on_grid[k], optimize=True)

    return _format_jks(vj_at_kpts_on_gpu, dm_at_kpts, kpts_band, kpts)


def get_k_kpts(df_object, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), kpts_band=None):
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
    coords = cupy.asarray(df_object.grids.coords)

    density_matrices_at_kpts = cupy.asarray(dm_kpts)
    formatted_density_matrices = _format_dms(density_matrices_at_kpts, kpts)
    n_channels, n_k_points, n_ao = formatted_density_matrices.shape[:3]
    data_type = df_object.ao_on_grid[-1].dtype

    weight = 1. / n_k_points * (cell.vol / df_object.n_grid_points)

    vk = cupy.zeros((n_channels, n_k_points, n_ao, n_ao), dtype=data_type)

    for k2_index, k2, ao_at_k2 in zip(range(n_k_points), kpts, df_object.ao_on_grid):
        density_dot_ao_at_k2 = cupy.ndarray((n_channels, n_ao, df_object.n_grid_points), dtype=data_type)
        for i in range(n_channels):
            density_dot_ao_at_k2[i] = formatted_density_matrices[i, k2_index] @ ao_at_k2.conj().T
        for k1_index, k1, ao_at_k1 in zip(range(n_k_points), kpts, df_object.ao_on_grid):
            e_ikr = cupy.exp(coords @ cupy.asarray(1j * (k1 - k2)))
            ao_sandwiched_e_ikr_in_real_space = cupy.einsum(
                "np, nq, n -> pqn", ao_at_k1.conj(), ao_at_k2, e_ikr).reshape(n_ao, n_ao, *mesh)

            ao_sandwiched_e_ikr_in_g_space = cupy.fft.fftn(ao_sandwiched_e_ikr_in_real_space,
                                                           axes=(2, 3, 4))

            coulomb_in_g_space = cupy.asarray(
                tools.get_coulG(df_object.cell, k=k2 - k1, mf=df_object, mesh=mesh).reshape(*mesh))

            coulomb_in_real_space = cupy.fft.ifftn(coulomb_in_g_space * ao_sandwiched_e_ikr_in_g_space,
                                                   axes=(2, 3, 4)).reshape(n_ao, n_ao, -1)

            for i in range(n_channels):
                coulomb_weighted_density_dot_ao_at_k2 = cupy.einsum("pqn, qn -> pn", coulomb_in_real_space,
                                                                    density_dot_ao_at_k2[i])

                coulomb_weighted_density_dot_ao_at_k2 *= e_ikr.conj()
                vk[i, k1_index, :, :] += weight * coulomb_weighted_density_dot_ao_at_k2 @ ao_at_k1

    return _format_jks(vk, dm_kpts, None, kpts)


class FFTDF(cpu_FFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        cpu_FFTDF.__init__(self, cell, kpts)
        self.to_gpu()
        numerical_integrator = self._numint
        self.ao_on_grid = [cupy.asarray(ao_at_k) for ao_at_k in
                           numerical_integrator.eval_ao(cell, self.grids.coords, kpts)]

        self.coulomb_in_k_space = cupy.asarray(tools.get_coulG(cell, mesh=self.mesh)).reshape(*self.mesh)
        self.n_grid_points = math.prod(self.mesh)
        self.exxdiv = None

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        j_on_gpu = get_j_kpts(self, dm, hermi, kpts, kpts_band)
        k_on_gpu = get_k_kpts(self, dm, hermi, kpts, kpts_band)

        return j_on_gpu.get(), k_on_gpu.get()
