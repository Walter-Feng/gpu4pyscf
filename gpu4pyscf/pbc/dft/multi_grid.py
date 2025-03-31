import ctypes

import numpy as np
import cupy as cp
import scipy

import pyscf.pbc.gto as gto

from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int

import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc import tools
import gpu4pyscf.pbc.dft.multigrid as multigrid_qiming
import gpu4pyscf.lib.cupy_helper as cupy_helper

libgpbc = cupy_helper.load_library("libgpbc")


def cast_to_pointer(array):
    if isinstance(array, cp.ndarray):
        return ctypes.cast(array.data.ptr, ctypes.c_void_p)
    elif isinstance(array, np.ndarray):
        return array.ctypes.data_as(ctypes.c_void_p)
    else:
        raise ValueError("Invalid array type")


def screen_gaussian_pairs(
    i_angular,
    j_angular,
    i_shells,
    j_shells,
    vectors_to_neighboring_images,
    mesh,
    atm,
    bas,
    env,
    threshold_in_log,
):
    n_i_shells = len(i_shells)
    n_j_shells = len(j_shells)
    n_images = len(vectors_to_neighboring_images)
    max_n_pairs = n_i_shells * n_j_shells * n_images * n_images
    non_trivial_pairs = cp.full(max_n_pairs, -1, dtype=cp.int32)
    image_indices = cp.full(max_n_pairs, -1, dtype=cp.int32)
    pairs_to_blocks_begin = cp.full((3, max_n_pairs), -1, dtype=cp.int32)
    pairs_to_blocks_end = cp.full((3, max_n_pairs), -1, dtype=cp.int32)
    libgpbc.screen_gaussian_pairs(
        cast_to_pointer(non_trivial_pairs),
        cast_to_pointer(image_indices),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        ctypes.c_int(i_angular),
        ctypes.c_int(j_angular),
        cast_to_pointer(i_shells),
        ctypes.c_int(n_i_shells),
        cast_to_pointer(j_shells),
        ctypes.c_int(n_j_shells),
        cast_to_pointer(vectors_to_neighboring_images),
        ctypes.c_int(len(vectors_to_neighboring_images)),
        (ctypes.c_int * 3)(*mesh),
        cast_to_pointer(atm),
        cast_to_pointer(bas),
        cast_to_pointer(env),
        ctypes.c_double(threshold_in_log),
    )
    non_trivial_index = cp.where(non_trivial_pairs >= 0)[0]
    non_trivial_pairs = non_trivial_pairs[non_trivial_index]
    image_indices = image_indices[non_trivial_index]
    pairs_to_blocks_begin = pairs_to_blocks_begin.T[non_trivial_index]
    pairs_to_blocks_end = pairs_to_blocks_end.T[non_trivial_index]
    return non_trivial_pairs, image_indices, pairs_to_blocks_begin, pairs_to_blocks_end


def assign_pairs_to_blocks(
    pairs_to_blocks_begin, pairs_to_blocks_end, n_blocks_abc, n_pairs, n_indices
):
    sorted_pairs_per_block = np.full(n_indices, -1, dtype=np.int32)
    n_pairs_per_block = np.full(np.prod(n_blocks_abc), -1, dtype=np.int32)
    block_index = np.full(np.prod(n_blocks_abc), -1, dtype=np.int32)
    libgpbc.assign_pairs_to_blocks(
        cast_to_pointer(sorted_pairs_per_block),
        cast_to_pointer(n_pairs_per_block),
        cast_to_pointer(block_index),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_pairs),
    )
    non_trivial_blocks = np.where(n_pairs_per_block > 0)[0]
    n_pairs_per_block = n_pairs_per_block[non_trivial_blocks]
    prepended_n_pairs_per_block = np.insert(n_pairs_per_block, 0, 0)
    accumulated_n_pairs_per_block = np.cumsum(
        prepended_n_pairs_per_block, dtype=np.int32
    )
    block_index = block_index[non_trivial_blocks]
    return (
        cp.asarray(sorted_pairs_per_block),
        cp.asarray(accumulated_n_pairs_per_block),
        cp.asarray(block_index),
    )


def assign_pairs_to_blocks_new(
    pairs_to_blocks_begin, pairs_to_blocks_end, n_blocks_abc, n_pairs, n_indices
):
    n_blocks = np.prod(n_blocks_abc)
    n_pairs_on_blocks = cp.full(n_blocks + 1, 0, dtype=cp.int32)
    libgpbc.count_pairs_on_blocks(
        cast_to_pointer(n_pairs_on_blocks),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_pairs),
    )
    n_contributing_blocks = int(n_pairs_on_blocks[-1])
    n_pairs_on_blocks = n_pairs_on_blocks[:-1]
    sorted_block_index = cp.asarray(cp.argsort(-n_pairs_on_blocks), dtype=cp.int32)
    n_pairs_on_blocks = n_pairs_on_blocks[sorted_block_index]
    accumulated_n_pairs_per_block = cp.full(n_blocks + 1, 0, dtype=cp.int32)
    accumulated_n_pairs_per_block[1:] = cp.cumsum(n_pairs_on_blocks, dtype=cp.int32)
    sorted_block_index = sorted_block_index[:n_contributing_blocks]
    pairs_on_blocks = cp.full(n_indices, -1, dtype=cp.int32)
    libgpbc.put_pairs_on_blocks(
        cast_to_pointer(pairs_on_blocks),
        cast_to_pointer(accumulated_n_pairs_per_block),
        cast_to_pointer(sorted_block_index),
        cast_to_pointer(pairs_to_blocks_begin),
        cast_to_pointer(pairs_to_blocks_end),
        cast_to_pointer(n_blocks_abc),
        ctypes.c_int(n_contributing_blocks),
        ctypes.c_int(n_pairs),
    )
    assert cp.all(pairs_on_blocks >= 0)
    return (
        cp.asarray(pairs_on_blocks),
        cp.asarray(accumulated_n_pairs_per_block),
        cp.asarray(sorted_block_index),
    )


def sort_gaussian_pairs(mydf, xc_type="LDA"):
    log = logger.new_logger(mydf, mydf.verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    vol = cell.vol
    block_size = np.array([4, 4, 4])
    lattice_vectors = cp.asarray(cell.lattice_vectors())
    off_diagonal = lattice_vectors - cp.diag(lattice_vectors.diagonal())
    is_non_orthogonal = cp.any(cp.abs(off_diagonal) > 1e-10)
    if is_non_orthogonal:
        is_non_orthogonal = 1
    else:
        is_non_orthogonal = 0
    reciprocal_lattice_vectors = cp.asarray(cp.linalg.inv(lattice_vectors).T, order="C")

    reciprocal_norms = cp.linalg.norm(reciprocal_lattice_vectors, axis=1)
    libgpbc.update_lattice_vectors(
        ctypes.cast(lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_norms.data.ptr, ctypes.c_void_p),
    )

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    t0 = log.timer("task generation", *t0)

    pairs = []
    for grids_localized, grids_diffused in tasks:
        subcell_in_localized_region = grids_localized.cell
        mesh = grids_localized.mesh
        n_blocks_abc = np.asarray(np.ceil(mesh / block_size), dtype=cp.int32)
        equivalent_cell_in_localized, coeff_in_localized = (
            subcell_in_localized_region.decontract_basis(to_cart=True, aggregate=True)
        )

        n_primitive_gtos_in_localized = multigrid._pgto_shells(
            subcell_in_localized_region
        )

        vectors_to_neighboring_images = cp.asarray(
            gto.eval_gto.get_lattice_Ls(subcell_in_localized_region)
        )
        phase_diff_among_images = cp.exp(
            1j
            * cp.asarray(mydf.kpts.reshape(-1, 3)).dot(vectors_to_neighboring_images.T)
        )
        if grids_diffused is None:
            grouped_cell = equivalent_cell_in_localized
            concatenated_coeff = scipy.linalg.block_diag(coeff_in_localized)
        else:
            subcell_in_diffused_region = grids_diffused.cell
            equivalent_cell_in_diffused, coeff_in_diffused = (
                subcell_in_diffused_region.decontract_basis(
                    to_cart=True, aggregate=True
                )
            )
            grouped_cell = equivalent_cell_in_localized + equivalent_cell_in_diffused
            concatenated_coeff = scipy.linalg.block_diag(
                coeff_in_localized, coeff_in_diffused
            )
        n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
        weight_penalty = np.prod(grouped_cell.mesh) / vol
        minimum_exponent = np.hstack(grouped_cell.bas_exps()).min()
        theta_ij = minimum_exponent / 2
        lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)

        precision = cell.precision / weight_penalty / lattice_summation_factor
        if xc_type != "LDA":
            precision *= 0.1
        threshold_in_log = np.log(precision * multigrid.EXTRA_PREC)

        shell_to_ao_indices = gto.moleintor.make_loc(grouped_cell._bas, "cart")
        per_angular_pairs = []

        i_angulars = grouped_cell._bas[:n_primitive_gtos_in_localized, multigrid.ANG_OF]
        i_angulars_unique = np.unique(i_angulars)
        sorted_i_shells = []
        for l in i_angulars_unique:
            i_shells = cp.asarray(np.where(i_angulars == l)[0], dtype=cp.int32)
            sorted_i_shells.append(i_shells)

        j_angulars = grouped_cell._bas[
            :n_primitive_gtos_in_two_regions, multigrid.ANG_OF
        ]
        j_angulars_unique = np.unique(j_angulars)
        sorted_j_shells = []
        for l in j_angulars_unique:
            j_shells = cp.asarray(np.where(j_angulars == l)[0], dtype=cp.int32)
            sorted_j_shells.append(j_shells)

        atm = cp.asarray(grouped_cell._atm, dtype=cp.int32)
        bas = cp.asarray(grouped_cell._bas, dtype=cp.int32)
        env = cp.asarray(grouped_cell._env)

        for i_angular, i_shells in zip(i_angulars_unique, sorted_i_shells):
            for j_angular, j_shells in zip(j_angulars_unique, sorted_j_shells):
                t1 = log.init_timer()
                i_angular = int(i_angular)
                j_angular = int(j_angular)
                (
                    shell_pair_indices_from_images,
                    image_indices,
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                ) = screen_gaussian_pairs(
                    i_angular,
                    j_angular,
                    i_shells,
                    j_shells,
                    vectors_to_neighboring_images,
                    mesh,
                    atm,
                    bas,
                    env,
                    threshold_in_log,
                )
                contributing_block_ranges = (
                    pairs_to_blocks_end - pairs_to_blocks_begin + 1
                )
                n_contributing_blocks_per_pair = cp.prod(
                    contributing_block_ranges, axis=1
                )
                n_indices = int(cp.sum(n_contributing_blocks_per_pair))
                n_pairs = len(shell_pair_indices_from_images)
                """ pairs_to_blocks_begin_cpu = pairs_to_blocks_begin.get()
                pairs_to_blocks_end_cpu = pairs_to_blocks_end.get()
                (
                    gaussian_pair_indices,
                    accumulated_counts,
                    sorted_contributing_blocks,
                ) = assign_pairs_to_blocks(
                    pairs_to_blocks_begin_cpu,
                    pairs_to_blocks_end_cpu,
                    n_blocks_abc,
                    n_pairs,
                    n_indices,
                )"""
                (
                    gaussian_pair_indices,
                    accumulated_counts,
                    sorted_contributing_blocks,
                ) = assign_pairs_to_blocks_new(
                    pairs_to_blocks_begin,
                    pairs_to_blocks_end,
                    n_blocks_abc,
                    n_pairs,
                    n_indices,
                )
                per_angular_pairs.append(
                    {
                        "angular": (i_angular, j_angular),
                        "non_trivial_pairs": shell_pair_indices_from_images,
                        "non_trivial_pairs_at_local_points": cp.asarray(
                            gaussian_pair_indices, dtype=cp.int32
                        ),
                        "accumulated_n_pairs_per_point": cp.asarray(
                            accumulated_counts, dtype=cp.int32
                        ),
                        "sorted_block_index": cp.asarray(
                            sorted_contributing_blocks, dtype=cp.int32
                        ),
                        "image_indices": image_indices,
                        "i_shells": cp.asarray(i_shells, dtype=cp.int32),
                        "j_shells": cp.asarray(j_shells, dtype=cp.int32),
                        "shell_to_ao_indices": cp.asarray(
                            shell_to_ao_indices, dtype=cp.int32
                        ),
                    }
                )

        ao_indices_in_localized = cp.asarray(grids_localized.ao_idx)
        if grids_diffused is None:
            ao_indices_in_diffused = cp.array([], dtype=cp.int32)
        else:
            ao_indices_in_diffused = cp.asarray(grids_diffused.ao_idx)
        pairs.append(
            {
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": cp.asarray(vectors_to_neighboring_images),
                "grouped_cell": grouped_cell,
                "mesh": np.asarray(grids_localized.mesh, dtype=np.int32),
                "ao_indices_in_localized": ao_indices_in_localized,
                "ao_indices_in_diffused": ao_indices_in_diffused,
                "concatenated_ao_indices": cp.concatenate(
                    (ao_indices_in_localized, ao_indices_in_diffused)
                ),
                "coeff_in_localized": cp.asarray(coeff_in_localized),
                "concatenated_coeff": cp.asarray(concatenated_coeff),
                "atm": cp.asarray(grouped_cell._atm, dtype=cp.int32),
                "bas": cp.asarray(grouped_cell._bas, dtype=cp.int32),
                "env": cp.asarray(grouped_cell._env),
                "phase_diff_among_images": phase_diff_among_images,
                "dxyz_dabc": cp.asarray(
                    (lattice_vectors.T / cp.asarray(mesh)).T, order="C"
                ),
                "is_non_orthogonal": is_non_orthogonal,
            }
        )

    mydf.sorted_gaussian_pairs = pairs
    t0 = log.timer("sort_gaussian_pairs", *t0)


def evaluate_density_wrapper(pairs_info, dm_slice, ignore_imag=True):
    c_driver = libgpbc.evaluate_density_driver
    n_k_points, n_images = pairs_info["phase_diff_among_images"].shape
    if n_k_points == 0:
        density_matrix_with_translation = cp.repeat(dm_slice, n_images, axis=1)
    else:
        density_matrix_with_translation = cp.einsum(
            "kt, ikpq->itpq", pairs_info["phase_diff_among_images"], dm_slice
        )

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape

    if ignore_imag is False:
        raise NotImplementedError
    density_matrix_with_translation_real_part = (
        density_matrix_with_translation.real.flatten()
    )
    density = cp.zeros((n_channels,) + tuple(pairs_info["mesh"]))

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]

        c_driver(
            cast_to_pointer(density),
            cast_to_pointer(density_matrix_with_translation_real_part),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["non_trivial_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(
                gaussians_per_angular_pair["non_trivial_pairs_at_local_points"]
            ),
            cast_to_pointer(
                gaussians_per_angular_pair["accumulated_n_pairs_per_point"]
            ),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            (ctypes.c_int * 3)(*pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
        )

    return density


def evaluate_density_on_g_mesh(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0):
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_channels, n_k_points, nao = dms.shape[:3]

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        raise NotImplementedError

    assert deriv < 1
    # gga_high_order = False
    density_slices = 1  # Presumably
    xc_type = "LDA"

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros(
        (n_channels * density_slices, nx, ny, nz), dtype=cp.complex128
    )
    for pairs in mydf.sorted_gaussian_pairs:

        mesh = pairs["mesh"]
        fft_grids = list(
            map(
                lambda mesh_points: np.fft.fftfreq(
                    mesh_points, 1.0 / mesh_points
                ).astype(np.int32),
                mesh,
            )
        )
        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1.0 / n_k_points * mydf.cell.vol / n_grid_points

        density_matrix_with_rows_in_localized = dms[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["concatenated_ao_indices"],
        ]
        density_matrix_with_rows_in_diffused = dms[
            :,
            :,
            pairs["ao_indices_in_diffused"][:, None],
            pairs["ao_indices_in_localized"],
        ]

        if deriv == 0:
            n_ao_in_diffused, n_ao_in_localized = (
                density_matrix_with_rows_in_diffused.shape[2:]
            )
            density_matrix_with_rows_in_localized[
                :, :, :, n_ao_in_localized:
            ] += density_matrix_with_rows_in_diffused.transpose(0, 1, 3, 2)

            coeff_sandwiched_density_matrix = cp.einsum(
                "nkij,pi->nkpj",
                density_matrix_with_rows_in_localized,
                pairs["coeff_in_localized"],
            )

            coeff_sandwiched_density_matrix = cp.einsum(
                "nkpj, qj -> nkpq",
                coeff_sandwiched_density_matrix,
                pairs["concatenated_coeff"],
            )

            libgpbc.update_dxyz_dabc(
                ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
            )

            density = evaluate_density_wrapper(pairs, coeff_sandwiched_density_matrix)
        else:
            raise NotImplementedError

        density_contribution_on_g_mesh = (
            tools.fft(density.reshape(n_channels * density_slices, -1), mesh)
            * weight_per_grid_point
        )

        density_on_g_mesh[
            cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)
        ] += density_contribution_on_g_mesh.reshape((-1,) + tuple(mesh))

    density_on_g_mesh = density_on_g_mesh.reshape([n_channels, density_slices, -1])
    return density_on_g_mesh


def evaluate_xc_wrapper(pairs_info, xc_weights, xc_type="LDA"):
    c_driver = libgpbc.evaluate_xc_driver
    n_i_functions = len(pairs_info["coeff_in_localized"])
    n_j_functions = len(pairs_info["concatenated_coeff"])

    n_channels = xc_weights.shape[0]
    n_k_points, n_images = pairs_info["phase_diff_among_images"].shape

    if xc_type != "LDA":
        raise NotImplementedError

    fock = cp.zeros((n_channels, n_images, n_i_functions, n_j_functions))
    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        c_driver(
            cast_to_pointer(fock),
            cast_to_pointer(xc_weights),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            cast_to_pointer(gaussians_per_angular_pair["non_trivial_pairs"]),
            cast_to_pointer(gaussians_per_angular_pair["i_shells"]),
            cast_to_pointer(gaussians_per_angular_pair["j_shells"]),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            cast_to_pointer(gaussians_per_angular_pair["shell_to_ao_indices"]),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            cast_to_pointer(
                gaussians_per_angular_pair["non_trivial_pairs_at_local_points"]
            ),
            cast_to_pointer(
                gaussians_per_angular_pair["accumulated_n_pairs_per_point"]
            ),
            cast_to_pointer(gaussians_per_angular_pair["sorted_block_index"]),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            cast_to_pointer(gaussians_per_angular_pair["image_indices"]),
            cast_to_pointer(pairs_info["neighboring_images"]),
            ctypes.c_int(n_images),
            cast_to_pointer(pairs_info["mesh"]),
            cast_to_pointer(pairs_info["atm"]),
            cast_to_pointer(pairs_info["bas"]),
            cast_to_pointer(pairs_info["env"]),
            ctypes.c_int(n_channels),
            ctypes.c_int(pairs_info["is_non_orthogonal"]),
        )

    if n_k_points > 1:
        return cp.einsum(
            "kt, ntij -> nkij", pairs_info["phase_diff_among_images"], fock
        )
    else:
        return cp.sum(fock, axis=1).reshape(
            n_channels, n_k_points, n_i_functions, n_j_functions
        )


def convert_xc_on_g_mesh_to_fock(
    mydf, xc_on_g_mesh, hermi=1, kpts=np.zeros((1, 3)), verbose=None
):
    cell = mydf.cell
    n_k_points = len(kpts)
    nao = cell.nao_nr()
    xc_on_g_mesh = xc_on_g_mesh.reshape(-1, *mydf.mesh)
    n_channels = xc_on_g_mesh.shape[0]
    at_gamma_point = multigrid.gamma_point(kpts)

    if hermi != 1:
        raise NotImplementedError

    data_type = cp.float64
    if not at_gamma_point:
        data_type = cp.complex128

    fock = cp.zeros((n_channels, n_k_points, nao, nao), dtype=data_type)

    for pairs in mydf.sorted_gaussian_pairs:
        mesh = pairs["mesh"]
        n_grid_points = np.prod(mesh)

        fft_grids = map(
            lambda mesh_points: np.fft.fftfreq(mesh_points, 1.0 / mesh_points).astype(
                np.int32
            ),
            mesh,
        )
        interpolated_xc_on_g_mesh = xc_on_g_mesh[
            cp.ix_(cp.arange(xc_on_g_mesh.shape[0]), *fft_grids)
        ].reshape(n_channels, n_grid_points)
        reordered_xc_on_real_mesh = tools.ifft(interpolated_xc_on_g_mesh, mesh).reshape(
            n_channels, n_grid_points
        )
        # order='C' forces a copy. otherwise the array is not contiguous
        reordered_xc_on_real_mesh = cp.asarray(
            reordered_xc_on_real_mesh.real, order="C"
        )
        n_ao_in_localized = len(pairs["ao_indices_in_localized"])
        libgpbc.update_dxyz_dabc(
            ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
        )
        fock_slice = evaluate_xc_wrapper(pairs, reordered_xc_on_real_mesh, "LDA")
        fock_slice = cp.einsum("nkpq,pi->nkiq", fock_slice, pairs["coeff_in_localized"])
        fock_slice = cp.einsum("nkiq,qj->nkij", fock_slice, pairs["concatenated_coeff"])

        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_localized"],
        ] += fock_slice[:, :, :, :n_ao_in_localized]
        fock[
            :,
            :,
            pairs["ao_indices_in_localized"][:, None],
            pairs["ao_indices_in_diffused"],
        ] += fock_slice[:, :, :, n_ao_in_localized:]
        if hermi == 1:
            fock[
                :,
                :,
                pairs["ao_indices_in_diffused"][:, None],
                pairs["ao_indices_in_localized"],
            ] += (
                fock_slice[:, :, :, n_ao_in_localized:].transpose(0, 1, 3, 2).conj()
            )
        else:
            raise NotImplementedError

    return fock


def get_nuc(mydf, kpts=None):
    kpts, is_single_kpt = fft._check_kpts(mydf, kpts)
    cell = mydf.cell
    mesh = mydf.mesh
    charge = cp.asarray(-cell.atom_charges())
    Gv = cell.get_Gv(mesh)
    SI = cp.asarray(cell.get_SI(Gv))
    rhoG = cp.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    hermi = 1
    vne = convert_xc_on_g_mesh_to_fock(mydf, vneG, hermi, kpts)[0]

    if is_single_kpt:
        vne = vne[0]
    return vne


def get_pp(mydf, kpts=None):
    """Get the periodic pseudopotential nuc-el AO matrix, with G=0 removed."""
    from pyscf import gto

    assert kpts is None or all(kpts == 0)
    is_single_kpt = False
    if kpts is None or kpts.ndim == 1:
        is_single_kpt = True
    kpts = np.zeros((1, 3))

    cell = mydf.cell
    log = logger.new_logger(cell)
    t0 = log.init_timer()
    mesh = mydf.mesh
    # Compute the vpplocG as
    # -einsum('ij,ij->j', pseudo.get_vlocG(cell, Gv), cell.get_SI(Gv))
    vpplocG = multigrid_qiming.eval_vpplocG(cell, mesh)
    vpp = convert_xc_on_g_mesh_to_fock(mydf, vpplocG, hermi=1, kpts=kpts)[0]
    t1 = log.timer_debug1("vpploc", *t0)

    vppnl = pp_int.get_pp_nl(cell, kpts)
    for k, kpt in enumerate(kpts):
        if is_single_kpt:
            vpp[k] += cp.asarray(vppnl[k].real)
        else:
            vpp[k] += cp.asarray(vppnl[k])

    if is_single_kpt:
        vpp = vpp[0]
    log.timer_debug1("vppnl", *t1)
    log.timer("get_pp", *t0)
    return vpp


def nr_rks(
    mydf,
    xc_code,
    dm_kpts,
    hermi=1,
    kpts=None,
    kpts_band=None,
    with_j=False,
    return_j=False,
    verbose=None,
):
    if kpts is None:
        kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    t0 = log.init_timer()
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order="C")
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset = dms.shape[0]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    numerical_integrator = mydf._numint
    xc_type = numerical_integrator._xc_type(xc_code)

    if xc_type == "LDA":
        derivative_order = 0
    else:
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    density_on_G_mesh = evaluate_density_on_g_mesh(
        mydf, dm_kpts, hermi, kpts, derivative_order
    )
    coulomb_on_g_mesh = cp.einsum(
        "ng,g->ng", density_on_G_mesh[:, 0], mydf.coulomb_kernel_on_g_mesh
    )
    coulomb_energy = 0.5 * cp.einsum(
        "ng,ng->n", density_on_G_mesh[:, 0].real, coulomb_on_g_mesh.real
    )
    coulomb_energy += 0.5 * cp.einsum(
        "ng,ng->n", density_on_G_mesh[:, 0].imag, coulomb_on_g_mesh.imag
    )
    coulomb_energy /= cell.vol

    log.debug2("Multigrid Coulomb energy %s", coulomb_energy)
    t0 = log.timer("coulomb", *t0)
    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density_in_real_space = tools.ifft(
        density_on_G_mesh.reshape(-1, ngrids), mesh
    ).real * (1.0 / weight)
    density_in_real_space = density_in_real_space.reshape(nset, -1, ngrids)
    n_electrons = density_in_real_space[:, 0].sum(axis=1) * weight
    weighted_xc_for_fock_on_g_mesh = cp.ndarray(
        (nset, *density_in_real_space.shape), dtype=cp.complex128
    )
    xc_energy_sum = np.zeros(nset)
    for i in range(nset):
        if xc_type == "LDA":
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(
                xc_code, density_in_real_space[i, 0], deriv=1, xctype=xc_type
            )[:2]
        else:
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(
                xc_code, density_in_real_space[i], deriv=1, xctype=xc_type
            )[:2]

        xc_energy_sum[i] += (
            density_in_real_space[i, 0] * xc_for_energy.flatten()
        ).sum() * weight

        weighted_xc_for_fock_on_g_mesh[i] = tools.fft(xc_for_fock * weight, mesh)
    density_in_real_space = density_on_G_mesh = None
    if nset == 1:
        coulomb_energy = coulomb_energy[0]
        n_electrons = n_electrons[0]
        xc_energy_sum = xc_energy_sum[0]
    log.debug("Multigrid exc %s  nelec %s", xc_energy_sum, n_electrons)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xc_type == "LDA":
        if with_j:
            weighted_xc_for_fock_on_g_mesh[:, 0] += coulomb_on_g_mesh
        xc_for_fock = convert_xc_on_g_mesh_to_fock(
            mydf, weighted_xc_for_fock_on_g_mesh, hermi, kpts_band
        )

    else:
        raise NotImplementedError
    t0 = log.timer("xc", *t0)

    shape = dm_kpts.shape
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    xc_for_fock = xc_for_fock.reshape(shape)
    xc_for_fock = cupy_helper.tag_array(
        xc_for_fock, ecoul=coulomb_energy, exc=xc_energy_sum, vj=None, vk=None
    )
    return n_electrons, xc_energy_sum, xc_for_fock


class FFTDF(fft.FFTDF, multigrid.MultiGridFFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        self.sorted_gaussian_pairs = None
        fft.FFTDF.__init__(self, cell, kpts)
        sort_gaussian_pairs(self)
        self.coulomb_kernel_on_g_mesh = tools.get_coulG(cell, mesh=self.mesh)

    get_nuc = get_nuc
    get_pp = get_pp


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell, kpts=mf.kpts), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
