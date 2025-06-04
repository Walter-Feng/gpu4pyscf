from ase.build import bulk
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import numpy as np
from pyscf.pbc import gto, tools
from pyscf.pbc import dft as cpu_pbcdft
from pyscf.pbc.dft import multigrid as cpu_multi_grid
from pyscf.pbc.grad import rhf as cpu_grad

from gpu4pyscf.pbc import dft as gpu_pbcdft
from gpu4pyscf.pbc.dft import multi_grid as gpu_multi_grid_mine
from gpu4pyscf.pbc.grad import rhf as gpu_grad

water_labels = ["32", "64", "128", "256", "512", "1024", "2048", "4096"]
cell_sizes = [9.8528, 12.4138, 15.6404, 19.7340, 24.8630, 31.3250, 39.4680, 49.7260]


def make_water_cluster(ke_cutoff, precision, index):

    water_label = water_labels[index]
    xyz_file = f"{water_label}.xyz"
    cell_size = cell_sizes[index]
    lattice_vectors = np.array(
        [[cell_size, 0, 0], [0, cell_size, 0], [0, 0, cell_size]]
    )

    cell = gto.M(
        a=lattice_vectors,
        atom=xyz_file,
        basis="gth-tzv2p",
        verbose=5,
        unit="aa",
        ke_cutoff=ke_cutoff,
        precision=precision,
        pseudo="gth-pade",
    )
    cell.exp_to_discard = 0.1
    cell.build()

    return cell


def make_multi_grid_object(mf, version):
    if version == "Rui":
        return gpu_multi_grid_mine.fftdf(mf)
    elif version == "Xing_CPU":
        mf.with_df = cpu_multi_grid.MultiGridFFTDF2(mf.cell)
        return mf
    elif version == "GDF_CPU":
        raise NotImplementedError("GDF_CPU is not implemented")

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

def make_gradient(mf, version):
    if "CPU" in version:
        return cpu_grad.Gradients(mf)
    else:
        return gpu_grad.Gradients(mf)

def print_snipping_line():
    print()
    print("-" * 30, "snip me", "-" * 30)
    print()


def print_header(name):
    print("=" * 100)
    print(name)
    print("=" * 100)


def run(ke_cutoff, precision, index, xc, version):
    print_snipping_line()
    label = water_labels[index]
    print_header(
        f"ke_cutoff={ke_cutoff}, precision={precision}, N={label}, xc={xc}, version={version}"
    )
    cell = make_water_cluster(ke_cutoff, precision, index)
    mf = make_mf(cell, xc, version)
    mf.kernel()
    grad = make_gradient(mf, version)
    grad.kernel()

# warmup
run(140, 1e-6, 0, "LDA", "Rui")

print_header("warmup done")
print_snipping_line()

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
        versions = ["Xing_CPU"]
    else:
        versions = ["Rui"]
    for index in range(len(water_labels)-6):
        for precision in [1e-6]:
            for ke_cutoff in [140]:
                for version in versions:
                    for xc in ["LDA", "PBE"]:
                        run(ke_cutoff, precision, index, xc, version)
