import cupy as cp
from pyscf.pbc.df.df_jk import _format_kpts_band

from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc import dft
from gpu4pyscf.pbc.df.fft import _check_kpts
from gpu4pyscf.pbc.df.fft_jk import _format_dms, _format_jks
from gpu4pyscf.pbc.dft import multigrid_v2
from gpu4pyscf.qmmm import QMMMSCF


class MultigridNumInt(multigrid_v2.MultiGridNumInt):
    def __init__(self, cell, external_potential):
        multigrid_v2.MultiGridNumInt.__init__(self, cell)
        self.fft_external_potential = external_potential.flatten()
        multigrid_v2.fft_in_place(self.fft_external_potential)

    def nr_rks(
        self,
        cell,
        grids,
        xc_code,
        dm_kpts,
        relativity=0,
        hermi=1,
        kpts=None,
        kpts_band=None,
        with_j=False,
        verbose=None,
    ):

        n, exc, vxc = multigrid_v2.MultiGridNumInt.nr_rks(
            self,
            cell,
            grids,
            xc_code,
            dm_kpts,
            relativity,
            hermi,
            kpts,
            kpts_band,
            with_j,
            verbose,
        )

        kpts, _ = _check_kpts(self, kpts)

        mm_fock_contribution = multigrid_v2.convert_xc_on_g_mesh_to_fock(
            self, self.fft_external_potential, kpts=kpts
        )

        formatted_dm_kpts = _format_dms(dm_kpts, kpts)

        energy_correction = contract(
            "nkij, nkij -> n", formatted_dm_kpts, mm_fock_contribution
        )

        kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

        vxc += _format_jks(mm_fock_contribution, dm_kpts, input_band, kpts)
        exc += energy_correction

        return n, exc, vxc
