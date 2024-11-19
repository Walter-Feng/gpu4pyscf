from pyscf.pbc.df.fft import FFTDF as cpu_FFTDF
import numpy as np


class FFTDF(cpu_FFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        cpu_FFTDF.__init__(self, cell, kpts)
        self.to_gpu()
        print("gpu object initialized")

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        print("GPU object called")

        quit()


