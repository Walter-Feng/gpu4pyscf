from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from pyscf.pbc import dft as cpu_pbcdft
from pyscf.pbc.dft import multigrid as cpu_multi_grid

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.dft import multigrid as gpu_multi_grid_qiming


def make_diamond_cell(ke_cutoff, precision, super_cell):
    diamond_cell = bulk("C", "diamond", orthorhombic=True)

    cell = gto.M(
        h=np.array(diamond_cell.cell),
        atom=ase_atoms_to_pyscf(diamond_cell),
        basis="gth-tzv2p",
        verbose=6,
        unit="aa",
        ke_cutoff=ke_cutoff,
        precision=precision,
        pseudo="gth-pade",
    )
    cell.exp_to_discard = 0.1

    cell = tools.super_cell(cell, super_cell)
    cell.build()
    return cell


def make_multi_grid_object(mf, version):
    if version == "Rui":
        return gpu_multi_grid_mine.fftdf(mf)
    elif version == "Qiming":
        mf._numint = gpu_multi_grid_qiming.MultiGridNumInt(mf.cell)
        return mf
    elif version == "Qiming_CPU":
        mf.with_df = cpu_multi_grid.MultiGridFFTDF(mf.cell)
        return mf
    elif version == "Xing_CPU":
        mf.with_df = cpu_multi_grid.MultiGridFFTDF2(mf.cell)
        return mf

    raise ValueError(f"Unknown version: {version}")


def make_mf(cell, xc, version):
    mf = None
    if "CPU" in version:
        mf = cpu_pbcdft.RKS(cell)
    else:
        mf = gpu_pbcdft.RKS(cell)
    mf.xc = xc
    mf.init_guess = "atom"
    mf = make_multi_grid_object(mf, version)
    return mf


def print_snipping_line():
    print()
    print("-" * 30, "snip me", "-" * 30)
    print()


def print_header(name):
    print("=" * 100)
    print(name)
    print("=" * 100)


def run(ke_cutoff, precision, super_cell, xc, version):
    print_snipping_line()
    print_header(
        f"ke_cutoff={ke_cutoff}, precision={precision}, super_cell={super_cell}, xc={xc}, version={version}"
    )
    cell = make_diamond_cell(ke_cutoff, precision, super_cell)
    mf = make_mf(cell, xc, version)
    mf.kernel()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run DFT calculations on diamond structure"
    )
    parser.add_argument(
        "platform",
        choices=["cpu", "gpu"],
        help="Choose between CPU or GPU implementation",
    )
    args = parser.parse_args()

    if args.platform == "cpu":
        versions = ["Qiming_CPU", "Xing_CPU"]
    else:
        versions = ["Rui", "Qiming"]
    for super_cell in [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]:
        for precision in [1e-6, 1e-8, 1e-10]:
            for ke_cutoff in [100, 150, 200]:
                for version in versions:
                    for xc in ["LDA", "PBE"]:
                        run(ke_cutoff, precision, super_cell, xc, version)
