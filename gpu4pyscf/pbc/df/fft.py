from pyscf.pbc.df.fft import FFTDF as cpu_FFTDF
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc import tools
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero
import numpy as np
import cupy
import math

def get_j_kpts(df_object, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None, weights = None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

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

    # Maybe the ao_on_grid of shape (n_channels, n_ao, n_grid_points) cannot be saved.
    density_in_real_space = cupy.einsum('ikpq, knp, knq->in', formatted_density_matrices, df_object.ao_on_grid.conj(), df_object.ao_on_grid)
    density_in_real_space /= n_k_points
    density_in_real_space = density_in_real_space.reshape(-1, *mesh)
    density_in_k_space = cupy.fft.fftn(density_in_real_space, axes=(1,2,3))
    coulomb_weighted_density_in_k_space = cupy.einsum("xyz, ixyz -> ixyz", df_object.coulomb_in_k_space, density_in_k_space)
    coulomb_in_real_space = cupy.fft.ifftn(coulomb_weighted_density_in_k_space, axes=(1,2,3)).reshape(n_channels, -1)

    if hermi == 1 or is_zero(kpts):
        coulomb_in_real_space = coulomb_in_real_space.real

    formatted_kpts_band = _format_kpts_band(kpts_band, kpts)

    n_bands = len(formatted_kpts_band)
    weight = cell.vol / df_object.n_grid_points
    coulomb_in_real_space *= weight

    vj_at_kpts_on_gpu = cupy.einsum('knp, in, knq -> ikpq', df_object.ao_on_grid.conj(), coulomb_in_real_space, df_object.ao_on_grid)

    return _format_jks(vj_at_kpts_on_gpu, dm_kpts, kpts_band, kpts)


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
        
        get_j_kpts(self, dm, hermi, kpts, kpts_band)

        quit()


