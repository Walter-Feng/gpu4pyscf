import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.pbc.dft.multigrid as multigrid_qiming
import gpu4pyscf.lib.cupy_helper as cupy_helper

import pyscf.pbc.gto as gto
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf import lib as cpu_lib
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import ft_ao
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band
from pyscf.pbc.gto.pseudo import pp_int

import numpy as np
import cupy as cp
import scipy

import ctypes

libgpbc = cupy_helper.load_library("libgpbc")

PTR_COORD = 1
EIJ_CUTOFF = 60


def CINTcommon_fac_sp(angular):
    if angular == 0:
        return 0.282094791773878143
    if angular == 1:
        return 0.488602511902919921
    else:
        return 1


def gaussian_summation_cutoff(exponents, angular, prefactors, threshold_in_log):
    prefactors_in_log = cp.log(cp.abs(prefactors))
    l = angular + 1
    r_reference = 10
    log_r = cp.log(r_reference)
    log_of_doubled_exponents = cp.log(2 * exponents)
    approximated_log_of_sum = (l + 1) * log_r - log_of_doubled_exponents
    branched_indices = cp.where(2 * log_r + log_of_doubled_exponents > 1)[0]
    if branched_indices.size > 0:
        approximated_log_of_sum[branched_indices] = (
            -(l + 4) // 2 * log_of_doubled_exponents[branched_indices]
        )
    approximated_log_of_sum += prefactors_in_log - threshold_in_log
    another_estimate_indices = cp.where(approximated_log_of_sum < exponents)[0]
    if another_estimate_indices.size > 0:
        approximated_log_of_sum[another_estimate_indices] = (
            prefactors_in_log[another_estimate_indices] - threshold_in_log
        )
    return cp.sqrt(cp.clip(approximated_log_of_sum, 0, None) / exponents)


def sort_gaussian_pairs(mydf, xc_type="LDA", blocking_sizes=np.array([4, 4, 4])):
    log = logger.new_logger(mydf, mydf.verbose)
    t0 = log.init_timer()
    blocking_sizes_on_gpu = cp.asarray(blocking_sizes)
    cell = mydf.cell
    lattice_vectors = cp.asarray(cell.lattice_vectors())
    reciprocal_lattice_vectors = cp.asarray(cp.linalg.inv(lattice_vectors).T, order="C")

    reciprocal_norms = cp.linalg.norm(reciprocal_lattice_vectors, axis=1)
    numerical_integrator = mydf._numint
    libgpbc.update_lattice_vectors(
        ctypes.cast(lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_lattice_vectors.data.ptr, ctypes.c_void_p),
        ctypes.cast(reciprocal_norms.data.ptr, ctypes.c_void_p),
    )

    tasks = getattr(mydf, "tasks", None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    pairs = []

    for grids_dense, grids_sparse in tasks:
        subcell_in_dense_region = grids_dense.cell

        if grids_sparse is None:
            ao_values = numerical_integrator.eval_ao(
                grids_dense.cell, grids_dense.coords, mydf.kpts
            )

            pairs.append({"mesh": grids_dense.mesh, "ao_values": ao_values})

        else:
            mesh = cp.asarray(grids_dense.mesh)
            granularized_mesh_size = cp.asarray(
                cp.ceil(mesh / blocking_sizes_on_gpu), dtype=cp.int32
            ).get()

            subcell_in_sparse_region = grids_sparse.cell
            equivalent_cell_in_dense, coeff_in_dense = (
                subcell_in_dense_region.decontract_basis(to_cart=True, aggregate=True)
            )
            equivalent_cell_in_sparse, coeff_in_sparse = (
                subcell_in_sparse_region.decontract_basis(to_cart=True, aggregate=True)
            )
            grouped_cell = equivalent_cell_in_dense + equivalent_cell_in_sparse
            concatenated_coeff = scipy.linalg.block_diag(
                coeff_in_dense, coeff_in_sparse
            )

            vol = grouped_cell.vol
            weight_penalty = np.prod(grouped_cell.mesh) / vol
            minimum_exponent = np.hstack(grouped_cell.bas_exps()).min()
            theta_ij = minimum_exponent / 2
            lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)

            precision = (
                grouped_cell.precision / weight_penalty / lattice_summation_factor
            )
            if xc_type != "LDA":
                precision *= 0.1
            threshold_in_log = np.log(precision * multigrid.EXTRA_PREC)
            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
            vectors_to_neighboring_images = cp.asarray(
                gto.eval_gto.get_lattice_Ls(grouped_cell)
            )
            n_images = len(vectors_to_neighboring_images)
            phase_diff_among_images = cp.exp(
                1j
                * cp.asarray(mydf.kpts.reshape(-1, 3)).dot(
                    vectors_to_neighboring_images.T
                )
            )
            shell_to_ao_indices = gto.moleintor.make_loc(grouped_cell._bas, "cart")
            per_angular_pairs = []

            for i_angular in set(
                grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF]
            ):
                i_shells = np.where(
                    grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF]
                    == i_angular
                )[0]
                n_i_shells = len(i_shells)
                i_basis = grouped_cell._bas[i_shells]

                i_exponents = cp.asarray(
                    grouped_cell._env[i_basis[:, multigrid.PTR_EXP]]
                )
                i_coord_pointers = np.asarray(
                    grouped_cell._atm[i_basis[:, multigrid.ATOM_OF], PTR_COORD]
                )
                i_x = cp.asarray(grouped_cell._env[i_coord_pointers])
                i_y = cp.asarray(grouped_cell._env[i_coord_pointers + 1])
                i_z = cp.asarray(grouped_cell._env[i_coord_pointers + 2])
                i_coeffs = cp.asarray(
                    grouped_cell._env[i_basis[:, multigrid.PTR_COEFF]]
                ) * CINTcommon_fac_sp(i_angular)

                for j_angular in set(
                    grouped_cell._bas[
                        0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF
                    ]
                ):
                    j_shells = np.where(
                        grouped_cell._bas[
                            0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF
                        ]
                        == j_angular
                    )[0]
                    n_j_shells = len(j_shells)
                    j_basis = grouped_cell._bas[j_shells]

                    j_exponents = cp.tile(
                        cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_EXP]]),
                        n_images,
                    )
                    j_coord_pointers = np.array(
                        grouped_cell._atm[j_basis[:, multigrid.ATOM_OF], PTR_COORD]
                    )
                    j_x = cp.add.outer(
                        vectors_to_neighboring_images[:, 0],
                        cp.asarray(grouped_cell._env[j_coord_pointers]),
                    ).flatten()
                    j_y = cp.add.outer(
                        vectors_to_neighboring_images[:, 1],
                        cp.asarray(grouped_cell._env[j_coord_pointers + 1]),
                    ).flatten()
                    j_z = cp.add.outer(
                        vectors_to_neighboring_images[:, 2],
                        cp.asarray(grouped_cell._env[j_coord_pointers + 2]),
                    ).flatten()
                    j_coeffs = cp.tile(
                        cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_COEFF]]),
                        n_images,
                    ) * CINTcommon_fac_sp(j_angular)
                    pair_exponents = cp.add.outer(i_exponents, j_exponents).flatten()
                    multiplied = cp.outer(i_exponents, j_exponents).flatten()
                    exponent_in_prefactor = multiplied / pair_exponents
                    pair_coefficients = cp.outer(i_coeffs, j_coeffs).flatten()
                    j_image_indices = cp.repeat(
                        cp.arange(n_images, dtype=cp.int32), n_j_shells
                    )
                    gaussian_pair_indices = cp.add.outer(
                        cp.arange(n_i_shells) * n_j_shells,
                        cp.tile(cp.arange(n_j_shells), n_images),
                    ).flatten()

                    shell_pair_indices_from_images = []
                    image_indices = []
                    cutoffs = []
                    contributing_block_begin = []
                    contributing_block_end = []

                    for image_index, i_image in enumerate(
                        vectors_to_neighboring_images
                    ):
                        shifted_i_x = i_x + i_image[0]
                        shifted_i_y = i_y + i_image[1]
                        shifted_i_z = i_z + i_image[2]

                        interatomic_distance = cp.square(
                            cp.subtract.outer(shifted_i_x, j_x)
                        )
                        interatomic_distance += cp.square(
                            cp.subtract.outer(shifted_i_y, j_y)
                        )
                        interatomic_distance += cp.square(
                            cp.subtract.outer(shifted_i_z, j_z)
                        )
                        exponents = (
                            exponent_in_prefactor * interatomic_distance.flatten()
                        )

                        non_trivial_pairs = cp.where(exponents < EIJ_CUTOFF)[0]

                        if len(non_trivial_pairs) == 0:
                            continue

                        prefactors = (
                            cp.exp(-exponents[non_trivial_pairs])
                            * pair_coefficients[non_trivial_pairs]
                        )
                        selected_pair_exponents = pair_exponents[non_trivial_pairs]
                        gaussian_cutoffs = gaussian_summation_cutoff(
                            selected_pair_exponents,
                            i_angular + j_angular,
                            prefactors,
                            threshold_in_log,
                        )
                        non_trivial_cutoffs = cp.where(gaussian_cutoffs > 0)[0]

                        if len(non_trivial_cutoffs) == 0:
                            continue

                        gaussian_cutoffs = gaussian_cutoffs[non_trivial_cutoffs]
                        non_trivial_pairs = non_trivial_pairs[non_trivial_cutoffs]
                        selected_pair_exponents = selected_pair_exponents[
                            non_trivial_cutoffs
                        ]
                        pair_x = cp.add.outer(
                            i_exponents * shifted_i_x, j_exponents * j_x
                        ).flatten()[non_trivial_pairs]
                        pair_y = cp.add.outer(
                            i_exponents * shifted_i_y, j_exponents * j_y
                        ).flatten()[non_trivial_pairs]
                        pair_z = cp.add.outer(
                            i_exponents * shifted_i_z, j_exponents * j_z
                        ).flatten()[non_trivial_pairs]
                        centers_in_fractional = reciprocal_lattice_vectors.dot(
                            cp.vstack((pair_x, pair_y, pair_z))
                            / selected_pair_exponents
                        )

                        cutoffs_in_fractional = cp.outer(
                            reciprocal_norms, gaussian_cutoffs
                        )
                        begin = cp.asarray(
                            cp.ceil(
                                (centers_in_fractional - cutoffs_in_fractional).T * mesh
                            ),
                            dtype=cp.int32,
                        )
                        end = cp.asarray(
                            cp.floor(
                                (centers_in_fractional + cutoffs_in_fractional).T * mesh
                            ),
                            dtype=cp.int32,
                        )

                        broad_enough_pairs = cp.where(cp.all(begin <= end, axis=1))[0]
                        in_range_pairs = broad_enough_pairs[
                            cp.where(cp.all(begin[broad_enough_pairs] < mesh, axis=1))[
                                0
                            ]
                        ]
                        in_range_pairs = in_range_pairs[
                            cp.where(cp.all(end[in_range_pairs] >= 0, axis=1))[0]
                        ]
                        non_trivial_pairs = non_trivial_pairs[in_range_pairs]
                        begin = begin[in_range_pairs]
                        begin[begin < 0] = 0
                        begin //= blocking_sizes_on_gpu
                        end = end[in_range_pairs]
                        end -= mesh
                        end[end >= 0] = 0
                        end += mesh
                        end //= blocking_sizes_on_gpu
                        gaussian_cutoffs = gaussian_cutoffs[in_range_pairs]

                        contributing_block_begin.append(begin)
                        contributing_block_end.append(end)
                        cutoffs.append(gaussian_cutoffs)
                        shell_pair_indices_from_images.append(
                            gaussian_pair_indices[non_trivial_pairs]
                        )
                        tiled_image_indices = (
                            cp.tile(j_image_indices, n_i_shells)
                            + n_images * image_index
                        )
                        image_indices.append(tiled_image_indices[non_trivial_pairs])

                    shell_pair_indices_from_images = cp.asarray(
                        cp.concatenate(shell_pair_indices_from_images), dtype=cp.int32
                    )
                    cutoffs = cp.concatenate(cutoffs)
                    image_indices = cp.concatenate(image_indices)
                    contributing_block_begin = cp.concatenate(contributing_block_begin)
                    contributing_block_end = cp.concatenate(contributing_block_end)
                    contributing_block_ranges = (
                        contributing_block_end - contributing_block_begin
                    )
                    n_contributing_blocks_per_pair = cp.prod(
                        contributing_block_ranges, axis=1
                    )
                    sort_index = cp.argsort(-n_contributing_blocks_per_pair)

                    shell_pair_indices_from_images = shell_pair_indices_from_images[
                        sort_index
                    ]
                    image_indices = image_indices[sort_index]
                    contributing_block_begin = contributing_block_begin[sort_index]
                    contributing_block_end = contributing_block_end[sort_index]
                    contributing_indices_for_pairs = []
                    contributing_indices_per_pair = []
                    n_pairs = len(shell_pair_indices_from_images)
                    contributing_block_begin_cpu = contributing_block_begin.get()
                    contributing_block_end_cpu = contributing_block_end.get()
                    for i_pair in range(n_pairs):
                        begin = contributing_block_begin_cpu[i_pair]
                        end = contributing_block_end_cpu[i_pair]
                        ranges = end - begin + 1
                        contributing_local_grid = np.add.outer(
                            np.linspace(begin[0], end[0], ranges[0], dtype=np.int32)
                            * granularized_mesh_size[1]
                            * granularized_mesh_size[2],
                            np.add.outer(
                                np.linspace(begin[1], end[1], ranges[1], dtype=np.int32)
                                * granularized_mesh_size[2],
                                np.linspace(
                                    begin[2], end[2], ranges[2], dtype=np.int32
                                ),
                            ),
                        )
                        contributing_indices_for_pairs.append(
                            contributing_local_grid.flatten()
                        )
                        contributing_indices_per_pair.append(
                            np.full(
                                len(contributing_local_grid.flatten()),
                                i_pair,
                                dtype=np.int32,
                            )
                        )

                    contributing_indices_for_pairs = np.concatenate(
                        contributing_indices_for_pairs
                    )
                    sort_indices = np.argsort(contributing_indices_for_pairs)
                    gaussian_pair_indices = np.concatenate(
                        contributing_indices_per_pair
                    )[sort_indices]
                    contributing_indices_for_pairs = contributing_indices_for_pairs[
                        sort_indices
                    ]
                    contributing_blocks, counts = np.unique(
                        contributing_indices_for_pairs, return_counts=True
                    )
                    count_sort_index = np.argsort(-counts)
                    counts_per_block = np.zeros(
                        np.prod(granularized_mesh_size) + 1, dtype=np.int32
                    )
                    counts_per_block[contributing_blocks + 1] = counts
                    accumulated_counts = np.cumsum(counts_per_block)
                    sorted_contributing_blocks = contributing_blocks[count_sort_index]

                    per_angular_pairs.append(
                        {
                            "angular": (i_angular, j_angular),
                            "non_trivial_pairs": shell_pair_indices_from_images,
                            "cutoffs": cutoffs,
                            "contributing_area_begin": contributing_block_begin,
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

            pairs.append(
                {
                    "per_angular_pairs": per_angular_pairs,
                    "neighboring_images": cp.asarray(vectors_to_neighboring_images),
                    "grouped_cell": grouped_cell,
                    "mesh": grids_dense.mesh,
                    "ao_indices_in_dense": cp.asarray(grids_dense.ao_idx),
                    "ao_indices_in_sparse": cp.asarray(grids_sparse.ao_idx),
                    "concatenated_ao_indices": cp.concatenate(
                        (
                            cp.asarray(grids_dense.ao_idx),
                            cp.asarray(grids_sparse.ao_idx),
                        )
                    ),
                    "coeff_in_dense": cp.asarray(coeff_in_dense),
                    "concatenated_coeff": cp.asarray(concatenated_coeff),
                    "atm": cp.asarray(grouped_cell._atm, dtype=cp.int32),
                    "bas": cp.asarray(grouped_cell._bas, dtype=cp.int32),
                    "env": cp.asarray(grouped_cell._env),
                    "blocking_sizes": blocking_sizes,
                    "phase_diff_among_images": phase_diff_among_images,
                    "dxyz_dabc": cp.asarray((lattice_vectors.T / mesh).T, order="C"),
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
            ctypes.cast(density.data.ptr, ctypes.c_void_p),
            ctypes.cast(
                density_matrix_with_translation_real_part.data.ptr, ctypes.c_void_p
            ),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            ctypes.cast(
                gaussians_per_angular_pair["non_trivial_pairs"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["cutoffs"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(
                gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(
                gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p
            ),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            ctypes.cast(
                gaussians_per_angular_pair["shell_to_ao_indices"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            ctypes.cast(
                gaussians_per_angular_pair[
                    "non_trivial_pairs_at_local_points"
                ].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["accumulated_n_pairs_per_point"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["sorted_block_index"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            ctypes.cast(
                gaussians_per_angular_pair["image_indices"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(pairs_info["neighboring_images"].data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_images),
            (ctypes.c_int * 3)(*pairs_info["mesh"]),
            ctypes.cast(pairs_info["atm"].data.ptr, ctypes.c_void_p),
            ctypes.cast(pairs_info["bas"].data.ptr, ctypes.c_void_p),
            ctypes.cast(pairs_info["env"].data.ptr, ctypes.c_void_p),
            (ctypes.c_int * 3)(*pairs_info["blocking_sizes"]),
            ctypes.c_int(n_channels),
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
    for (grids_dense, grids_sparse), pairs in zip(tasks, mydf.sorted_gaussian_pairs):

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

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            density = cp.zeros(
                (n_channels, density_slices, n_grid_points), dtype=cp.complex128
            )
            ao_indices_in_dense = grids_dense.ao_idx
            density_matrix_in_dense_region = dms[
                :, :, ao_indices_in_dense[:, None], ao_indices_in_dense
            ]
            for k in range(n_k_points):
                for i in range(n_channels):
                    if xc_type == "LDA":
                        ao_dot_dm = cp.dot(
                            pairs["ao_values"][k], density_matrix_in_dense_region[i, k]
                        )
                        density_subblock = cp.einsum(
                            "xi,xi->x", ao_dot_dm, pairs["ao_values"][k].conj()
                        )
                    else:
                        raise NotImplementedError
                    density[i, :] += density_subblock
            if hermi:
                density = density.real

        else:
            density_matrix_with_rows_in_dense = dms[
                :,
                :,
                pairs["ao_indices_in_dense"][:, None],
                pairs["concatenated_ao_indices"],
            ]
            density_matrix_with_rows_in_sparse = dms[
                :,
                :,
                pairs["ao_indices_in_sparse"][:, None],
                pairs["ao_indices_in_dense"],
            ]

            if deriv == 0:
                n_ao_in_sparse, n_ao_in_dense = (
                    density_matrix_with_rows_in_sparse.shape[2:]
                )
                density_matrix_with_rows_in_dense[
                    :, :, :, n_ao_in_dense:
                ] += density_matrix_with_rows_in_sparse.transpose(0, 1, 3, 2)

                coeff_sandwiched_density_matrix = cp.einsum(
                    "nkij,pi->nkpj",
                    density_matrix_with_rows_in_dense,
                    pairs["coeff_in_dense"],
                )

                coeff_sandwiched_density_matrix = cp.einsum(
                    "nkpj, qj -> nkpq",
                    coeff_sandwiched_density_matrix,
                    pairs["concatenated_coeff"],
                )

                libgpbc.update_dxyz_dabc(
                    ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
                )

                density = evaluate_density_wrapper(
                    pairs, coeff_sandwiched_density_matrix
                )
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
    n_i_functions = len(pairs_info["coeff_in_dense"])
    n_j_functions = len(pairs_info["concatenated_coeff"])

    n_channels = xc_weights.shape[0]
    n_k_points, n_images = pairs_info["phase_diff_among_images"].shape

    if xc_type != "LDA":
        raise NotImplementedError

    fock = cp.zeros((n_channels, n_images, n_i_functions, n_j_functions))
    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        c_driver(
            ctypes.cast(fock.data.ptr, ctypes.c_void_p),
            ctypes.cast(xc_weights.data.ptr, ctypes.c_void_p),
            ctypes.c_int(i_angular),
            ctypes.c_int(j_angular),
            ctypes.cast(
                gaussians_per_angular_pair["non_trivial_pairs"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["cutoffs"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(
                gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(
                gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p
            ),
            ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
            ctypes.cast(
                gaussians_per_angular_pair["shell_to_ao_indices"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.c_int(n_i_functions),
            ctypes.c_int(n_j_functions),
            ctypes.cast(
                gaussians_per_angular_pair[
                    "non_trivial_pairs_at_local_points"
                ].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["accumulated_n_pairs_per_point"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.cast(
                gaussians_per_angular_pair["sorted_block_index"].data.ptr,
                ctypes.c_void_p,
            ),
            ctypes.c_int(len(gaussians_per_angular_pair["sorted_block_index"])),
            ctypes.cast(
                gaussians_per_angular_pair["image_indices"].data.ptr, ctypes.c_void_p
            ),
            ctypes.cast(pairs_info["neighboring_images"].data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_images),
            (ctypes.c_int * 3)(*pairs_info["mesh"]),
            ctypes.cast(pairs_info["atm"].data.ptr, ctypes.c_void_p),
            ctypes.cast(pairs_info["bas"].data.ptr, ctypes.c_void_p),
            ctypes.cast(pairs_info["env"].data.ptr, ctypes.c_void_p),
            (ctypes.c_int * 3)(*pairs_info["blocking_sizes"]),
            ctypes.c_int(n_channels),
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

    for (grids_dense, grids_sparse), pairs in zip(
        mydf.tasks, mydf.sorted_gaussian_pairs
    ):
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

        if grids_sparse is None:
            ao_index_in_dense = grids_dense.ao_idx
            for k in range(n_k_points):
                for i in range(n_channels):
                    xc_sub_block = cp.einsum(
                        "gi, g, gj -> ij",
                        pairs["ao_values"][k].conj(),
                        reordered_xc_on_real_mesh[i],
                        pairs["ao_values"][k],
                    )
                    fock[
                        i, k, ao_index_in_dense[:, None], ao_index_in_dense
                    ] += xc_sub_block
            ao_values = ao_on_sliced_grid_in_dense = None
        else:
            n_ao_in_sparse = len(pairs["ao_indices_in_dense"])
            libgpbc.update_dxyz_dabc(
                ctypes.cast(pairs["dxyz_dabc"].data.ptr, ctypes.c_void_p)
            )
            fock_slice = evaluate_xc_wrapper(pairs, reordered_xc_on_real_mesh, "LDA")
            fock_slice = cp.einsum("nkpq,pi->nkiq", fock_slice, pairs["coeff_in_dense"])
            fock_slice = cp.einsum(
                "nkiq,qj->nkij", fock_slice, pairs["concatenated_coeff"]
            )

            fock[
                :,
                :,
                pairs["ao_indices_in_dense"][:, None],
                pairs["ao_indices_in_dense"],
            ] += fock_slice[:, :, :, :n_ao_in_sparse]
            fock[
                :,
                :,
                pairs["ao_indices_in_dense"][:, None],
                pairs["ao_indices_in_sparse"],
            ] += fock_slice[:, :, :, n_ao_in_sparse:]
            if hermi == 1:
                fock[
                    :,
                    :,
                    pairs["ao_indices_in_sparse"][:, None],
                    pairs["ao_indices_in_dense"],
                ] += (
                    fock_slice[:, :, :, n_ao_in_sparse:].transpose(0, 1, 3, 2).conj()
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

    coulomb_kernel_on_g_mesh = tools.get_coulG(cell, mesh=mesh)
    coulomb_on_g_mesh = cp.einsum(
        "ng,g->ng", density_on_G_mesh[:, 0], coulomb_kernel_on_g_mesh
    )
    coulomb_energy = 0.5 * cp.einsum(
        "ng,ng->n", density_on_G_mesh[:, 0].real, coulomb_on_g_mesh.real
    )
    coulomb_energy += 0.5 * cp.einsum(
        "ng,ng->n", density_on_G_mesh[:, 0].imag, coulomb_on_g_mesh.imag
    )
    coulomb_energy /= cell.vol

    log.debug("Multigrid Coulomb energy %s", coulomb_energy)
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

    get_nuc = get_nuc
    get_pp = get_pp


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell, kpts=mf.kpts), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
