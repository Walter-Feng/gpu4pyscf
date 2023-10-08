# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pyscf.dft import roks
from gpu4pyscf.dft import numint
from gpu4pyscf.scf.hf import _get_jk, _eigh
from gpu4pyscf.lib.utils import patch_cpu_kernel, to_cpu, to_gpu

class ROKS(roks.ROKS):
    to_cpu = to_cpu
    to_gpu = to_gpu

    def __init__(self, mol, xc='LDA,VWN'):
        super().__init__(mol, xc)
        self._numint = numint.NumInt()

    @property
    def device(self):
        return self._numint.device
    @device.setter
    def device(self, value):
        self._numint.device = value

    get_jk = patch_cpu_kernel(roks.ROKS.get_jk)(_get_jk)
    _eigh = patch_cpu_kernel(roks.ROKS._eigh)(_eigh)
