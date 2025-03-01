import pyscf.pbc.gto as gto
import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.lib.cupy_helper as cupy_helper

from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band

import numpy as np
import cupy as cp
import scipy

import ctypes

libgpbc = cupy_helper.load_library('libgpbc')

PTR_COORD = 1
EIJ_CUTOFF = 60


def CINTcommon_fac_sp(angular):
    if angular == 0:
        return 0.282094791773878143
    if angular == 1:
        return 0.488602511902919921
    else:
        return 1


def accumulate(list):
    out = [0]
    for i in range(len(list)):
        if i == 0:
            out.append(list[i])
        else:
            out.append(out[i] + list[i])
    return out


def gaussian_summation_cutoff(exponents, angular, prefactors, threshold_in_log):
    prefactors_in_log = cp.log(cp.abs(prefactors))
    l = angular + 1
    r_reference = 10
    log_r = cp.log(r_reference)
    log_of_doubled_exponents = cp.log(2 * exponents)
    approximated_log_of_sum = (l + 1) * log_r - log_of_doubled_exponents
    branched_indices = cp.where(2 * log_r + log_of_doubled_exponents > 1)[0]
    if branched_indices.size > 0:
        approximated_log_of_sum[branched_indices] = - (l + 4) // 2 * log_of_doubled_exponents[branched_indices]
    approximated_log_of_sum += prefactors_in_log - threshold_in_log
    another_estimate_indices = cp.where(approximated_log_of_sum < exponents)[0]
    if another_estimate_indices.size > 0:
        approximated_log_of_sum[another_estimate_indices] = prefactors_in_log[
                                                                another_estimate_indices] - threshold_in_log
    return cp.sqrt(cp.clip(approximated_log_of_sum, 0, None) / exponents)


def sort_gaussian_pairs(mydf, xc_type="LDA", blocking_sizes=np.array([4, 4, 4])):
    blocking_sizes_on_gpu = cp.asarray(blocking_sizes)
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    lattice_vectors = cp.asarray(cell.lattice_vectors().T)
    reciprocal_lattice_vectors = cp.linalg.inv(lattice_vectors.T)
    reciprocal_norms = cp.linalg.norm(reciprocal_lattice_vectors, axis=1)

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    pairs = []

    for grids_dense, grids_sparse in tasks:
        subcell_in_dense_region = grids_dense.cell

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.

            pairs.append({"mesh": grids_dense.mesh})

        else:
            mesh = cp.asarray(grids_dense.mesh)
            granularized_mesh_size = cp.asarray(cp.ceil(mesh / blocking_sizes_on_gpu), dtype=cp.int32).get()

            granularized_mesh_to_indices = cp.arange(np.prod(granularized_mesh_size)).reshape(granularized_mesh_size)
            subcell_in_sparse_region = grids_sparse.cell
            equivalent_cell_in_dense, coeff_in_dense = subcell_in_dense_region.decontract_basis(
                to_cart=True, aggregate=True)
            equivalent_cell_in_sparse, coeff_in_sparse = subcell_in_sparse_region.decontract_basis(
                to_cart=True, aggregate=True)
            grouped_cell = equivalent_cell_in_dense + equivalent_cell_in_sparse
            concatenated_coeff = scipy.linalg.block_diag(coeff_in_dense, coeff_in_sparse)

            vol = grouped_cell.vol
            weight_penalty = np.prod(grouped_cell.mesh) / vol
            minimum_exponent = np.hstack(grouped_cell.bas_exps()).min()
            theta_ij = minimum_exponent / 2
            lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)

            precision = grouped_cell.precision / weight_penalty / lattice_summation_factor
            if xc_type != 'LDA':
                precision *= .1
            threshold_in_log = np.log(precision * multigrid.EXTRA_PREC)
            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
            vectors_to_neighboring_images = gto.eval_gto.get_lattice_Ls(grouped_cell)
            phase_diff_among_images = cp.exp(
                1j * cp.asarray(mydf.kpts.reshape(-1, 3).dot(vectors_to_neighboring_images.T)))
            shell_to_ao_indices = gto.moleintor.make_loc(grouped_cell._bas, 'cart')
            per_angular_pairs = []

            for i_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF]):
                i_shells = np.where(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF] == i_angular)[0]
                i_basis = grouped_cell._bas[i_shells]

                i_exponents = cp.asarray(grouped_cell._env[i_basis[:, multigrid.PTR_EXP]])
                i_coord_pointers = np.asarray(grouped_cell._atm[i_basis[:, multigrid.ATOM_OF], PTR_COORD])
                i_x = cp.asarray(grouped_cell._env[i_coord_pointers])
                i_y = cp.asarray(grouped_cell._env[i_coord_pointers + 1])
                i_z = cp.asarray(grouped_cell._env[i_coord_pointers + 2])
                i_coeffs = cp.asarray(grouped_cell._env[i_basis[:, multigrid.PTR_COEFF]]) * CINTcommon_fac_sp(i_angular)

                for j_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF]):
                    j_shells = np.where(
                        grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF] == j_angular)[0]
                    j_basis = grouped_cell._bas[j_shells]

                    j_exponents = cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_EXP]])
                    j_coord_pointers = np.array(grouped_cell._atm[j_basis[:, multigrid.ATOM_OF], PTR_COORD])
                    j_x = cp.asarray(grouped_cell._env[j_coord_pointers])
                    j_y = cp.asarray(grouped_cell._env[j_coord_pointers + 1])
                    j_z = cp.asarray(grouped_cell._env[j_coord_pointers + 2])
                    j_coeffs = cp.asarray(
                        grouped_cell._env[j_basis[:, multigrid.PTR_COEFF]]) * CINTcommon_fac_sp(j_angular)
                    pair_exponents = cp.add.outer(i_exponents, j_exponents).flatten()
                    multiplied = cp.outer(i_exponents, j_exponents).flatten()
                    exponent_in_prefactor = multiplied / pair_exponents
                    pair_coefficients = cp.outer(i_coeffs, j_coeffs).flatten()

                    non_trivial_pairs_from_images = []
                    image_indices = []
                    cutoffs = []
                    contributing_area_begin = []
                    contributing_area_end = []

                    for image_index, i_image in enumerate(vectors_to_neighboring_images):
                        shifted_i_x = i_x - i_image[0]
                        shifted_i_y = i_y - i_image[1]
                        shifted_i_z = i_z - i_image[2]

                        interatomic_distance = cp.square(cp.subtract.outer(shifted_i_x, j_x))
                        interatomic_distance += cp.square(cp.subtract.outer(shifted_i_y, j_y))
                        interatomic_distance += cp.square(cp.subtract.outer(shifted_i_z, j_z))
                        exponents = exponent_in_prefactor * interatomic_distance.flatten()

                        non_trivial_pairs = cp.where(exponents < EIJ_CUTOFF)[0]

                        if len(non_trivial_pairs) == 0:
                            continue

                        prefactors = cp.exp(-exponents[non_trivial_pairs]) * pair_coefficients[non_trivial_pairs]
                        selected_pair_exponents = pair_exponents[non_trivial_pairs]
                        gaussian_cutoffs = gaussian_summation_cutoff(selected_pair_exponents,
                                                                     i_angular + j_angular,
                                                                     prefactors, threshold_in_log)
                        non_trivial_cutoffs = cp.where(gaussian_cutoffs > 0)[0]

                        if len(non_trivial_cutoffs) == 0:
                            continue

                        gaussian_cutoffs = gaussian_cutoffs[non_trivial_cutoffs]
                        non_trivial_pairs = non_trivial_pairs[non_trivial_cutoffs]
                        selected_pair_exponents = selected_pair_exponents[non_trivial_cutoffs]
                        pair_x = cp.add.outer(i_exponents * shifted_i_x, j_exponents * j_x).flatten()[non_trivial_pairs]
                        pair_y = cp.add.outer(i_exponents * shifted_i_y, j_exponents * j_y).flatten()[non_trivial_pairs]
                        pair_z = cp.add.outer(i_exponents * shifted_i_z, j_exponents * j_z).flatten()[non_trivial_pairs]
                        centers_in_fractional = reciprocal_lattice_vectors.dot(cp.vstack((pair_x, pair_y, pair_z)) / selected_pair_exponents)

                        cutoffs_in_fractional = cp.outer(reciprocal_norms, gaussian_cutoffs)
                        begin = cp.ceil((centers_in_fractional - cutoffs_in_fractional).T * mesh)
                        end = cp.floor((centers_in_fractional + cutoffs_in_fractional).T * mesh)

                        broad_enough_pairs = cp.where(cp.all(begin <= end, axis=1))[0]
                        non_trivial_pairs = non_trivial_pairs[broad_enough_pairs]
                        begin = begin[broad_enough_pairs]
                        end = end[broad_enough_pairs]
                        gaussian_cutoffs = gaussian_cutoffs[broad_enough_pairs]

                        contributing_area_begin.append(begin - blocking_sizes_on_gpu + 1)
                        contributing_area_end.append(end)
                        cutoffs.append(gaussian_cutoffs)
                        non_trivial_pairs_from_images.append(non_trivial_pairs)
                        image_indices.append(cp.full(len(non_trivial_pairs), image_index))

                    non_trivial_pairs_from_images = cp.asarray(
                        cp.concatenate(non_trivial_pairs_from_images), dtype=cp.int32)
                    cutoffs = cp.concatenate(cutoffs)
                    image_indices = cp.asarray(cp.concatenate(image_indices), dtype=cp.int32)
                    contributing_area_begin = cp.asarray(cp.concatenate(contributing_area_begin), dtype=cp.int32).T
                    contributing_area_end = cp.asarray(cp.concatenate(contributing_area_end), dtype=cp.int32).T
                    bin = cp.array([8, 8, 8])
                    binned_ranges = cp.asarray(cp.ceil((contributing_area_end - contributing_area_begin).T / bin), dtype=cp.int32) * bin
                    binned_ranges = binned_ranges.get()
                    indices = []
                    unique = np.unique(binned_ranges, axis=0)
                    for unique_range in unique:
                        indices.append(cp.asarray(np.where(np.all(binned_ranges == unique_range, axis=1))[0], dtype=cp.int32))
                    n_pairs = len(non_trivial_pairs_from_images)
                    contributing_indices = []
                    for xyz_index in range(3):
                        local_grid_size = granularized_mesh_size[xyz_index]
                        begin_translation = cp.asarray(cp.ceil(cp.subtract.outer(
                            cp.arange(local_grid_size) * blocking_sizes[
                                xyz_index], contributing_area_end[xyz_index]) /
                                                    mesh[xyz_index]), dtype=cp.int32)
                        end_translation = cp.asarray(cp.floor(cp.subtract.outer(cp.arange(local_grid_size) * blocking_sizes[xyz_index], contributing_area_begin[xyz_index]) / mesh[xyz_index]), dtype=cp.int32)
                        contributing = (begin_translation <= end_translation).T
                        contributing_indices.append(list(map(lambda bools: cp.where(bools)[0], contributing)))

                    contributing_indices_for_pairs = []

                    for i_pair in range(n_pairs):
                        contributing_local_grid = granularized_mesh_to_indices[cp.ix_(
                            contributing_indices[0][i_pair],
                            contributing_indices[1][i_pair],
                            contributing_indices[2][i_pair])]
                        contributing_indices_for_pairs.append(contributing_local_grid.flatten())

                    pair_indices = cp.concatenate(
                        [cp.full(len(contributing_indices_for_pairs[i]), i) for i in range(n_pairs)])
                    contributing_indices_for_pairs = cp.concatenate(contributing_indices_for_pairs)
                    sort_indices = cp.argsort(contributing_indices_for_pairs)
                    contributing_indices_for_pairs = contributing_indices_for_pairs[sort_indices]
                    non_trivial_pairs_at_local_points = cp.asarray(pair_indices[sort_indices], dtype=cp.int32)
                    effective_local_grid_points, counts = np.unique(contributing_indices_for_pairs.get(), return_counts=True)
                    n_pairs_per_point = cp.zeros(cp.prod(granularized_mesh_size) + 1)
                    n_pairs_per_point[effective_local_grid_points + 1] = counts
                    accumulated_n_pairs_per_point = cp.asarray(cp.ufunc.accumulate(cp.add, n_pairs_per_point), dtype=cp.int32)

                    print("non_trivial_pairs: ", len(non_trivial_pairs_from_images))
                    per_angular_pairs.append({
                        "angular": (i_angular, j_angular),
                        "non_trivial_pairs": non_trivial_pairs_from_images,
                        "cutoffs": cutoffs,
                        "contributing_area_begin": contributing_area_begin,
                        "non_trivial_pairs_at_local_points": non_trivial_pairs_at_local_points,
                        "accumulated_n_pairs_per_point": accumulated_n_pairs_per_point,
                        "image_indices": image_indices,
                        "i_shells": cp.asarray(i_shells, dtype=cp.int32),
                        "j_shells": cp.asarray(j_shells, dtype=cp.int32),
                        "shell_to_ao_indices": cp.asarray(shell_to_ao_indices, dtype=cp.int32),
                        "block_sizes": unique,
                        "indices_with_same_block": indices
                    })

            pairs.append({
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": cp.asarray(vectors_to_neighboring_images),
                "phase_diff_among_images": phase_diff_among_images,
                "grouped_cell": grouped_cell,
                "mesh": grids_dense.mesh,
                "ao_indices_in_dense": cp.asarray(grids_dense.ao_idx),
                "ao_indices_in_sparse": cp.asarray(grids_sparse.ao_idx),
                "concatenated_ao_indices": cp.concatenate(
                    (cp.asarray(grids_dense.ao_idx), cp.asarray(grids_sparse.ao_idx))),
                "coeff_in_dense": cp.asarray(coeff_in_dense),
                "concatenated_coeff": cp.asarray(concatenated_coeff),
                "atm": cp.asarray(grouped_cell._atm, dtype=cp.int32),
                "bas": cp.asarray(grouped_cell._bas, dtype=cp.int32),
                "env": cp.asarray(grouped_cell._env),
                "lattice_vectors": lattice_vectors,
                "reciprocal_lattice_vectors": reciprocal_lattice_vectors,
                "blocking_sizes": blocking_sizes
            })

    mydf.sorted_gaussian_pairs = pairs

    assert 0

def evaluate_density_wrapper(pairs_info, dm_slice, ignore_imag=True):
    c_driver = libgpbc.evaluate_density_driver
    n_k_points, n_images = pairs_info["phase_diff_among_images"].shape
    if n_k_points == 0:
        density_matrix_with_translation = cp.repeat(dm_slice, n_images, axis = 1)
    else:
        density_matrix_with_translation = cp.einsum("kt, ikpq->itpq", pairs_info["phase_diff_among_images"], dm_slice)

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape

    if ignore_imag is False:
        raise NotImplementedError
    density_matrix_with_translation_real_part = density_matrix_with_translation.real.flatten()
    density = cp.zeros((n_channels,) + tuple(pairs_info["mesh"]))

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
        c_driver(ctypes.cast(density.data.ptr, ctypes.c_void_p),
                 ctypes.cast(density_matrix_with_translation_real_part.data.ptr, ctypes.c_void_p),
                 ctypes.c_int(i_angular), ctypes.c_int(j_angular),
                 ctypes.cast(gaussians_per_angular_pair["non_trivial_pairs"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["cutoffs"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p),
                 ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
                 ctypes.cast(gaussians_per_angular_pair["shell_to_ao_indices"].data.ptr, ctypes.c_void_p),
                 ctypes.c_int(n_i_functions), ctypes.c_int(n_j_functions),
                 ctypes.cast(gaussians_per_angular_pair["non_trivial_pairs_at_local_points"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["accumulated_n_pairs_per_point"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["image_indices"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["neighboring_images"].data.ptr, ctypes.c_void_p), ctypes.c_int(n_images),
                 ctypes.cast(pairs_info["lattice_vectors"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["reciprocal_lattice_vectors"].data.ptr, ctypes.c_void_p),
                 (ctypes.c_int * 3)(*pairs_info["mesh"]),
                 ctypes.cast(pairs_info["atm"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["bas"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["env"].data.ptr, ctypes.c_void_p),
                 (ctypes.c_int * 3)(*pairs_info["blocking_sizes"]), ctypes.c_int(n_channels))

    return density

def evaluate_density_diffused_wrapper(pairs_info, dm_slice, ignore_imag=True):
    c_driver = libgpbc.evaluate_diffused_density_driver
    n_k_points, n_images = pairs_info["phase_diff_among_images"].shape
    if n_k_points == 0:
        density_matrix_with_translation = cp.repeat(dm_slice, n_images, axis = 1)
    else:
        density_matrix_with_translation = cp.einsum("kt, ikpq->itpq", pairs_info["phase_diff_among_images"], dm_slice)

    n_channels, _, n_i_functions, n_j_functions = density_matrix_with_translation.shape

    if ignore_imag is False:
        raise NotImplementedError
    density_matrix_with_translation_real_part = density_matrix_with_translation.real.flatten()
    density = cp.zeros((n_channels,) + tuple(pairs_info["mesh"]))

    for gaussians_per_angular_pair in pairs_info["per_angular_pairs"]:
        for blocking_size, corresponding_indices in zip(gaussians_per_angular_pair["block_sizes"], gaussians_per_angular_pair["indices_with_same_block"]):
            selected_pairs = gaussians_per_angular_pair["non_trivial_pairs"][corresponding_indices]
            selected_image_indices = gaussians_per_angular_pair["image_indices"][selected_pairs]
            selected_pairs_area_begin = gaussians_per_angular_pair["contributing_area_begin"][:, selected_pairs]

            assert selected_pairs.dtype == cp.int32
            assert selected_image_indices.dtype == cp.int32
            assert selected_pairs_area_begin.dtype == cp.int32

            (i_angular, j_angular) = gaussians_per_angular_pair["angular"]
            c_driver(ctypes.cast(density.data.ptr, ctypes.c_void_p),
                     ctypes.cast(density_matrix_with_translation_real_part.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(i_angular), ctypes.c_int(j_angular),
                     ctypes.cast(selected_pairs.data.ptr, ctypes.c_void_p),
                     ctypes.c_int(len(selected_pairs)),
                     ctypes.cast(gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p),
                     ctypes.cast(gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p),
                     ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
                     ctypes.cast(gaussians_per_angular_pair["shell_to_ao_indices"].data.ptr, ctypes.c_void_p),
                     ctypes.c_int(n_i_functions), ctypes.c_int(n_j_functions),
                     ctypes.cast(selected_pairs_area_begin.data.ptr, ctypes.c_void_p),
                     ctypes.cast(selected_image_indices.data.ptr, ctypes.c_void_p),
                     ctypes.cast(pairs_info["neighboring_images"].data.ptr, ctypes.c_void_p), ctypes.c_int(n_images),
                     ctypes.cast(pairs_info["lattice_vectors"].data.ptr, ctypes.c_void_p),
                     ctypes.cast(pairs_info["reciprocal_lattice_vectors"].data.ptr, ctypes.c_void_p),
                     (ctypes.c_int * 3)(*pairs_info["mesh"]),
                     ctypes.cast(pairs_info["atm"].data.ptr, ctypes.c_void_p),
                     ctypes.cast(pairs_info["bas"].data.ptr, ctypes.c_void_p),
                     ctypes.cast(pairs_info["env"].data.ptr, ctypes.c_void_p),
                     (ctypes.c_int * 3)(*blocking_size), ctypes.c_int(n_channels))

    return density

def evaluate_density_on_g_mesh(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0, rho_g_high_order=None):
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_channels, n_k_points, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        raise "Tasks should not be None. Let me fix later"

    assert (deriv < 1)
    gga_high_order = False
    density_slices = 1  # Presumably
    if deriv == 0:
        xc_type = 'LDA'

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros((n_channels * density_slices, nx, ny, nz), dtype=cp.complex128)
    for (grids_dense, grids_sparse), pairs in zip(tasks, mydf.sorted_gaussian_pairs):

        subcell_in_dense_region = grids_dense.cell
        mesh = pairs["mesh"]
        fft_grids = list(map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))
        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1. / n_k_points * mydf.cell.vol / n_grid_points

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            density = cp.zeros((n_channels, density_slices, n_grid_points), dtype=cp.complex128)
            ao_indices_in_dense = grids_dense.ao_idx
            density_matrix_in_dense_region = dms[:, :, ao_indices_in_dense[:, None], ao_indices_in_dense]
            for ao_on_sliced_grid_in_dense, grid_begin, grid_end in mydf.aoR_loop(grids_dense, kpts, deriv):
                ao_values, mask = ao_on_sliced_grid_in_dense[0], ao_on_sliced_grid_in_dense[2]
                for k in range(n_k_points):
                    for i in range(n_channels):
                        if xc_type == 'LDA':
                            ao_dot_dm = cp.dot(ao_values[k], density_matrix_in_dense_region[i, k])
                            density_subblock = cp.einsum('xi,xi->x', ao_dot_dm, ao_values[k].conj())
                        else:
                            density_subblock = numint.eval_rho(subcell_in_dense_region, ao_values[k],
                                                               density_matrix_in_dense_region[i, k],
                                                               mask, xc_type, hermi)
                        density[i, :, grid_begin:grid_end] += density_subblock
                ao_values = ao_on_sliced_grid_in_dense = ao_dot_dm = None
            if hermi:
                density = density.real

        else:
            density_matrix_with_rows_in_dense = dms[:, :, pairs["ao_indices_in_dense"][:, None],
                                                pairs["concatenated_ao_indices"]]
            density_matrix_with_rows_in_sparse = dms[:, :, pairs["ao_indices_in_sparse"][:, None],
                                                 pairs["ao_indices_in_dense"]]

            if deriv == 0:
                n_ao_in_sparse, n_ao_in_dense = density_matrix_with_rows_in_sparse.shape[2:]
                density_matrix_with_rows_in_dense[:, :, :,
                n_ao_in_dense:] += density_matrix_with_rows_in_sparse.transpose(0, 1, 3, 2)

                coeff_sandwiched_density_matrix = cp.einsum('nkij,pi->nkpj',
                                                            density_matrix_with_rows_in_dense,
                                                            pairs["coeff_in_dense"])

                coeff_sandwiched_density_matrix = cp.einsum("nkpj, qj -> nkpq",
                                                            coeff_sandwiched_density_matrix,
                                                            pairs["concatenated_coeff"])

                density = evaluate_density_wrapper(pairs, coeff_sandwiched_density_matrix)
            else:
                raise NotImplementedError

        density_contribution_on_g_mesh = tools.fft(density.reshape(n_channels * density_slices, -1),
                                                   mesh) * weight_per_grid_point

        density_on_g_mesh[
            cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)] += density_contribution_on_g_mesh.reshape(
            (-1,) + tuple(mesh))

    density_on_g_mesh = density_on_g_mesh.reshape(n_channels, density_slices, -1)
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
        c_driver(ctypes.cast(fock.data.ptr, ctypes.c_void_p),
                 ctypes.cast(xc_weights.data.ptr, ctypes.c_void_p),
                 ctypes.c_int(i_angular), ctypes.c_int(j_angular),
                 ctypes.cast(gaussians_per_angular_pair["non_trivial_pairs"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["cutoffs"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p),
                 ctypes.c_int(len(gaussians_per_angular_pair["j_shells"])),
                 ctypes.cast(gaussians_per_angular_pair["shell_to_ao_indices"].data.ptr, ctypes.c_void_p),
                 ctypes.c_int(n_i_functions), ctypes.c_int(n_j_functions),
                 ctypes.cast(gaussians_per_angular_pair["non_trivial_pairs_at_local_points"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["accumulated_n_pairs_per_point"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(gaussians_per_angular_pair["image_indices"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["neighboring_images"].data.ptr, ctypes.c_void_p), ctypes.c_int(n_images),
                 ctypes.cast(pairs_info["lattice_vectors"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["reciprocal_lattice_vectors"].data.ptr, ctypes.c_void_p),
                 (ctypes.c_int * 3)(*pairs_info["mesh"]),
                 ctypes.cast(pairs_info["atm"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["bas"].data.ptr, ctypes.c_void_p),
                 ctypes.cast(pairs_info["env"].data.ptr, ctypes.c_void_p),
                 (ctypes.c_int * 3)(*pairs_info["blocking_sizes"]), ctypes.c_int(n_channels))

    if n_k_points > 1:
        return cp.einsum("kt, ntij -> nkij", pairs_info["phase_diff_among_images"], fock)
    else:
        return cp.sum(fock, axis=1).reshape(n_channels, n_k_points, n_i_functions, n_j_functions)


def convert_xc_on_g_mesh_to_fock(mydf, xc_on_g_mesh, hermi=1, kpts=np.zeros((1, 3)), verbose=None):
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

    for (grids_dense, grids_sparse), pairs in zip(mydf.tasks, mydf.sorted_gaussian_pairs):
        mesh = pairs["mesh"]
        n_grid_points = np.prod(mesh)
        fft_grids = map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh)
        interpolated_xc_on_g_mesh = xc_on_g_mesh[
            cp.ix_(cp.arange(xc_on_g_mesh.shape[0]), *fft_grids)].reshape(n_channels, n_grid_points)

        reordered_xc_on_real_mesh = tools.ifft(interpolated_xc_on_g_mesh, mesh).reshape(n_channels, n_grid_points)
        # order='C' forces a copy. otherwise the array is not contiguous
        reordered_xc_on_real_mesh = cp.asarray(reordered_xc_on_real_mesh.real, order='C')

        if grids_sparse is None:
            ao_index_in_dense = grids_dense.ao_idx
            for ao_on_sliced_grid_in_dense, p0, p1 in mydf.aoR_loop(grids_dense, kpts):
                ao_values = ao_on_sliced_grid_in_dense[0]
                for k in range(n_k_points):
                    for i in range(n_channels):
                        xc_scaled_ao = numint._scale_ao(ao_values[k], reordered_xc_on_real_mesh[i, p0:p1])
                        xc_sub_block = cp.dot(ao_values[k].conj().T, xc_scaled_ao)
                        fock[i, k, ao_index_in_dense[:, None], ao_index_in_dense] += xc_sub_block
                ao_values = ao_on_sliced_grid_in_dense = None

        else:
            n_ao_in_sparse = len(pairs["ao_indices_in_dense"])

            fock_slice = evaluate_xc_wrapper(pairs, reordered_xc_on_real_mesh, "LDA")

            fock_slice = cp.einsum('nkpq,pi->nkiq', fock_slice, pairs["coeff_in_dense"])
            fock_slice = cp.einsum('nkiq,qj->nkij', fock_slice, pairs["concatenated_coeff"])

            fock[:, :, pairs["ao_indices_in_dense"][:, None], pairs["ao_indices_in_dense"]] += fock_slice[:, :, :,
                                                                                               :n_ao_in_sparse]
            fock[:, :, pairs["ao_indices_in_dense"][:, None], pairs["ao_indices_in_sparse"]] += fock_slice[:, :, :,
                                                                                                n_ao_in_sparse:]

            if hermi == 1:
                fock[:, :, pairs["ao_indices_in_sparse"][:, None], pairs["ao_indices_in_dense"]] += \
                    fock_slice[:, :, :, n_ao_in_sparse:].transpose(0, 1, 3, 2).conj()
            else:
                raise NotImplementedError

    return fock


def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if kpts is None: kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    numerical_integrator = mydf._numint
    xc_type = numerical_integrator._xc_type(xc_code)

    if xc_type == 'LDA':
        derivative_order = 0
    else:
        raise NotImplementedError

    mesh = mydf.mesh
    ngrids = np.prod(mesh)

    density_on_G_mesh = evaluate_density_on_g_mesh(mydf, dm_kpts, hermi, kpts, derivative_order)
    coulomb_kernel_on_g_mesh = tools.get_coulG(cell, mesh=mesh)
    coulomb_on_g_mesh = cp.einsum('ng,g->ng', density_on_G_mesh[:, 0], coulomb_kernel_on_g_mesh)
    coulomb_energy = .5 * cp.einsum('ng,ng->n', density_on_G_mesh[:, 0].real, coulomb_on_g_mesh.real)
    coulomb_energy += .5 * cp.einsum('ng,ng->n', density_on_G_mesh[:, 0].imag, coulomb_on_g_mesh.imag)
    coulomb_energy /= cell.vol
    log.debug('Multigrid Coulomb energy %s', coulomb_energy)

    weight = cell.vol / ngrids

    # *(1./weight) because rhoR is scaled by weight in _eval_rhoG.  When
    # computing rhoR with IFFT, the weight factor is not needed.
    density_in_real_space = tools.ifft(density_on_G_mesh.reshape(-1, ngrids), mesh).real * (1. / weight)
    density_in_real_space = density_in_real_space.reshape(nset, -1, ngrids)
    n_electrons = density_in_real_space[:, 0].sum(axis=1) * weight

    weighted_xc_for_fock_on_g_mesh = cp.ndarray((nset, *density_in_real_space.shape), dtype=cp.complex128)
    xc_energy_sum = cp.zeros(nset)
    for i in range(nset):
        if xc_type == 'LDA':
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i, 0], deriv=1,
                                                                          xctype=xc_type)[:2]
        else:
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i], deriv=1,
                                                                          xctype=xc_type)[:2]

        xc_energy_sum[i] += (density_in_real_space[i, 0] * xc_for_energy.flatten()).sum() * weight

        weighted_xc_for_fock_on_g_mesh[i] = tools.fft(xc_for_fock * weight, mesh)
    density_in_real_space = density_on_G_mesh = None

    if nset == 1:
        coulomb_energy = coulomb_energy[0]
        n_electrons = n_electrons[0]
        xc_energy_sum = xc_energy_sum[0]
    log.debug('Multigrid exc %s  nelec %s', xc_energy_sum, n_electrons)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xc_type == 'LDA':
        if with_j:
            weighted_xc_for_fock_on_g_mesh[:, 0] += coulomb_on_g_mesh

        xc_for_fock = convert_xc_on_g_mesh_to_fock(mydf, weighted_xc_for_fock_on_g_mesh, hermi, kpts_band)

    else:
        raise NotImplementedError

    if return_j:
        vj = convert_xc_on_g_mesh_to_fock(mydf, coulomb_on_g_mesh, hermi, kpts_band, verbose=log)
        vj = fft_jk._format_jks(vj, dm_kpts, input_band, kpts)
    else:
        vj = None

    shape = list(dm_kpts.shape)
    if len(shape) == 3 and shape[0] != kpts_band.shape[0]:
        shape[0] = kpts_band.shape[0]
    xc_for_fock = xc_for_fock.reshape(shape)
    xc_for_fock = cupy_helper.tag_array(xc_for_fock, ecoul=coulomb_energy, exc=xc_energy_sum, vj=vj, vk=None)
    return n_electrons, xc_energy_sum, xc_for_fock


class FFTDF(fft.FFTDF, multigrid.MultiGridFFTDF):
    def __init__(self, cell, kpts=np.zeros((1, 3))):
        self.sorted_gaussian_pairs = None
        fft.FFTDF.__init__(self, cell, kpts)
        sort_gaussian_pairs(self)


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
