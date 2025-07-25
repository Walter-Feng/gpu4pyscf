GPU plugin for PySCF
====================
![nightly](https://github.com/pyscf/gpu4pyscf/actions/workflows/nightly_build.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/gpu4pyscf-cuda11x.svg)](https://badge.fury.io/py/gpu4pyscf-cuda11x)

Installation
--------

> [!NOTE]
> The compiled binary packages support compute capability 7.0 and later (Volta and later, such as Tesla V100, RTX 20 series and later).

To check your installed CUDA Toolkit version, run
```sh
nvcc --version
```
Then, install the appropriate package based on your CUDA version:

| Platform      | Command                               | cutensor (**highly recommended**)|
----------------| --------------------------------------|----------------------------------|
| **CUDA 11.x** |  ```pip3 install gpu4pyscf-cuda11x``` | ```pip3 install cutensor-cu11``` |
| **CUDA 12.x** |  ```pip3 install gpu4pyscf-cuda12x``` | ```pip3 install cutensor-cu12``` |

The versions of CuPy and cuTENSOR are strongly interdependent and should not be combined arbitrarily.
The recommended combinations include:
1. CuPy 13.3.0 + cuTENSOR 2.0.2
2. CuPy 13.4.1 + cuTENSOR 2.2.0

Using other versions or combinations may lead to failures in functionality. 
We **recommend** creating a dedicated environment using:
```sh
pip3 install --no-cache-dir -r requirements.txt
```
This ensures compatibility and reproducibility, especially since this configuration is used in our nightly benchmarks.

Compilation
--------
To compile the package, run the following commands:
```sh
git clone https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
cmake --build build/temp.gpu4pyscf -j 4
CURRENT_PATH=`pwd`
export PYTHONPATH="${PYTHONPATH}:${CURRENT_PATH}"
```
Then install cutensor and cupy for acceleration (please switch the versions according to your runtime CUDA environment!)
```sh
pip3 install cutensor-cu12 cupy-cuda12x
```
There shouldn't be cupy or cutensor compilation during pip install process. If you see the following warning at the beginning of a gpu4pyscf job, it implies problems with cupy and cutensor installation (likely a version mismatch, or multiple versions of same package installed).
```
<repo_path>/gpu4pyscf/lib/cutensor.py:<line_number>: UserWarning: using cupy as the tensor contraction engine.
```

Features
--------
- Density fitting scheme and direct SCF scheme;
- SCF, analytical gradient, and analytical Hessian calculations for Hartree-Fock and DFT;
- LDA, GGA, mGGA, hybrid, and range-separated functionals via [libXC](https://gitlab.com/libxc/libxc/-/tree/master/);
- Spin-conserved and spin-flip TDA and TDDFT for excitated states;
- Geometry optimization and transition state search via [geomeTRIC](https://geometric.readthedocs.io/en/latest/);
- Atomic Simulation Environment ([ASE](https://gitlab.com/ase/ase)) interface;
- Dispersion corrections via [DFTD3](https://github.com/dftd3/simple-dftd3) and [DFTD4](https://github.com/dftd4/dftd4);
- Analytical gradient and analytical Hessian for nonlocal functional correction (vv10);
- GPU accelerated ECP;
- PCM models, their analytical gradients, and analytical Hessian matrix;
- SMD solvent model;
- Unrestricted Hartree-Fock and unrestricted DFT, gradient, and Hessian;
- CHELPG, ESP, and RESP atomic charge;

The following features are still in the experimental stage
- MP2/DF-MP2 and CCSD;
- Polarizability, IR, and NMR shielding;
- Raman spectrum;
- QM/MM with PBC;
- Multi-GPU for both direct SCF and density fitting;
- SCF and DFT with periodic boundary condition;
- Non-adiabatic coupling for TDDFT;
- Energy decomposition analysis;

Limitations
--------
- Rys roots up to 9 for density fitting scheme and direct scf scheme;
- Atomic basis up to g orbitals;
- Auxiliary basis up to i orbitals;
- Density fitting scheme up to ~168 atoms with def2-tzvpd basis, bounded by CPU memory;
- meta-GGA without density laplacian;
- Double hybrid functionals are not supported;
- Hessian of TDDFT is not supported;

Examples
--------
```python
import pyscf
from gpu4pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='LDA').density_fit()

e_dft = mf.kernel()  # compute total energy
print(f"total energy = {e_dft}")

g = mf.nuc_grad_method()
g_dft = g.kernel()   # compute analytical gradient

h = mf.Hessian()
h_dft = h.kernel()   # compute analytical Hessian

```

`to_gpu` is supported since PySCF 2.5.0
```python
import pyscf
from pyscf.dft import rks

atom ='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

mol = pyscf.M(atom=atom, basis='def2-tzvpp')
mf = rks.RKS(mol, xc='LDA').density_fit().to_gpu()  # move PySCF object to GPU4PySCF object
e_dft = mf.kernel()  # compute total energy

```

Find more examples in [gpu4pyscf/examples](https://github.com/pyscf/gpu4pyscf/tree/master/examples)

Benchmarks
--------
Speedup with GPU4PySCF v0.6.0 on A100-80G over Q-Chem 6.1 on 32-cores CPU (density fitting, SCF, def2-tzvpp, def2-universal-jkfit, B3LYP, (99,590))

| mol               |   natm |    LDA |    PBE |   B3LYP |    M06 |   wB97m-v |
|:------------------|-------:|-------:|-------:|--------:|-------:|----------:|
| 020_Vitamin_C     |     20 |   2.86 |   6.09 |   13.11 |  11.58 |     17.46 |
| 031_Inosine       |     31 |  13.14 |  15.87 |   16.57 |  25.89 |     26.14 |
| 033_Bisphenol_A   |     33 |  12.31 |  16.88 |   16.54 |  28.45 |     28.82 |
| 037_Mg_Porphin    |     37 |  13.85 |  19.03 |   20.53 |  28.31 |     30.27 |
| 042_Penicillin_V  |     42 |  10.34 |  13.35 |   15.34 |  22.01 |     24.2  |
| 045_Ochratoxin_A  |     45 |  13.34 |  15.3  |   19.66 |  27.08 |     25.41 |
| 052_Cetirizine    |     52 |  17.79 |  17.44 |   19    |  24.41 |     25.87 |
| 057_Tamoxifen     |     57 |  14.7  |  16.57 |   18.4  |  24.86 |     25.47 |
| 066_Raffinose     |     66 |  13.77 |  14.2  |   20.47 |  22.94 |     25.35 |
| 084_Sphingomyelin |     84 |  14.24 |  12.82 |   15.96 |  22.11 |     24.46 |
| 095_Azadirachtin  |     95 |   5.58 |   7.72 |   24.18 |  26.84 |     25.21 |
| 113_Taxol         |    113 |   5.44 |   6.81 |   24.58 |  29.14 |    nan    |

Find more benchmarks in [gpu4pyscf/benchmarks](https://github.com/pyscf/gpu4pyscf/tree/master/benchmarks)

References
---------
```
@misc{li2024introducting,
      title={Introducing GPU-acceleration into the Python-based Simulations of Chemistry Framework},
      author={Rui Li and Qiming Sun and Xing Zhang and Garnet Kin-Lic Chan},
      year={2024},
      eprint={2407.09700},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2407.09700},
}

@misc{wu2024enhancing,
      title={Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework},
      author={Xiaojie Wu and Qiming Sun and Zhichen Pu and Tianze Zheng and Wenzhi Ma and Wen Yan and Xia Yu and Zhengxiao Wu and Mian Huo and Xiang Li and Weiluo Ren and Sheng Gong and Yumin Zhang and Weihao Gao},
      year={2024},
      eprint={2404.09452},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2404.09452},
}
```
