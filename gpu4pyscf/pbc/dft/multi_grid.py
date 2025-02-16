import pyscf.pbc.gto as gto
import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.lib.cupy_helper as cupy_helper

from pyscf.dft.numint import libdft
from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band

import numpy as np
import cupy as cp
import scipy

import ctypes

libgdft = cupy_helper.load_library('libgdft')
libgpbc = cupy_helper.load_library('libgpbc')

PTR_COORD = 1
EIJ_CUTOFF = 60

new_intermediates = []
old_intermediates = []


def accumulate(list):
    out = [0]
    for i in range(len(list)):
        if i == 0:
            out.append(list[i])
        else:
            out.append(out[i] + list[i])
    return out


def eval_mat(cell, weights, shls_slice=None, comp=1,
             xctype='LDA', kpts=None, global_mesh=None, offset=None, local_mesh=None):
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if global_mesh is None:
        global_mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(global_mesh) / vol
    exp_min = np.hstack(cell.bas_exps()).min()
    theta_ij = exp_min / 2
    lattice_sum_fac = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_sum_fac
    if xctype != 'LDA':
        precision *= .1
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[multigrid.PTR_EXPDROP] = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shls_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    vectors_to_neighboring_images = gto.eval_gto.get_lattice_Ls(cell)
    nimgs = len(vectors_to_neighboring_images)

    xctype = xctype.upper()
    n_mat = None
    if xctype == 'LDA':
        if weights.ndim == 1:
            weights = weights.reshape(-1, np.prod(global_mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, np.prod(global_mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    lattice_vector = cell.lattice_vectors()
    b = np.linalg.inv(lattice_vector.T)
    if offset is None:
        offset = (0, 0, 0)
    if local_mesh is None:
        local_mesh = global_mesh
    # log_prec is used to estimate the gto_rcut. Add EXTRA_PREC to count
    # other possible factors and coefficients in the integral.
    log_prec = np.log(precision * multigrid.EXTRA_PREC)

    if abs(lattice_vector - np.diag(lattice_vector.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    eval_fn = 'NUMINTeval_' + xctype.lower() + lattice_type
    drv = libdft.NUMINT_fill2c

    atm_on_gpu = cp.asarray(atm)
    bas_on_gpu = cp.asarray(bas)
    env_on_gpu = cp.asarray(env)
    ao_loc_on_gpu = cp.asarray(ao_loc)
    vectors_to_neighboring_images_on_gpu = cp.asarray(vectors_to_neighboring_images)
    new_driver = libgpbc.evaluate_xc_driver

    def make_mat(xc_weights):
        fock = np.zeros((nimgs, comp, naoj, naoi))
        drv(getattr(libdft, eval_fn),
            xc_weights.get().ctypes.data_as(ctypes.c_void_p),
            fock.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(comp), ctypes.c_int(0),
            (ctypes.c_int * 4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            vectors_to_neighboring_images.ctypes.data_as(ctypes.c_void_p),
            lattice_vector.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*local_mesh),
            (ctypes.c_int * 3)(*global_mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(fock)

    def new_driver_wrapper(xc_weights):
        fock = cp.zeros((nimgs, comp, naoj, naoi))
        assert isinstance(xc_weights, cp.ndarray)
        new_driver(ctypes.cast(fock.data.ptr, ctypes.c_void_p),
                   ctypes.cast(xc_weights.data.ptr, ctypes.c_void_p),
                   (ctypes.c_int * 4)(i0, i1, j0, j1),
                   ctypes.cast(ao_loc_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.c_int(0), ctypes.c_int(0),
                   ctypes.c_int(nimgs),
                   ctypes.cast(vectors_to_neighboring_images_on_gpu.data.ptr, ctypes.c_void_p),
                   lattice_vector.ctypes.data_as(ctypes.c_void_p),
                   (ctypes.c_int * 3)(*offset),
                   (ctypes.c_int * 3)(*local_mesh),
                   (ctypes.c_int * 3)(*global_mesh),
                   ctypes.cast(atm_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(bas_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(env_on_gpu.data.ptr, ctypes.c_void_p))

        return fock

    out = []
    for wv in weights:
        if cell.dimension == 0:
            raise NotImplementedError
        elif kpts is None or multigrid.gamma_point(kpts):
            mat = new_driver_wrapper(wv).sum(axis=0).transpose(0, 2, 1)
            if comp == 1:
                mat = mat[0]
            if getattr(kpts, 'ndim', None) == 2:
                mat = mat[None, :]
        else:
            mat = new_driver_wrapper(wv)
            expkL = cp.exp(1j * kpts.reshape(-1, 3).dot(vectors_to_neighboring_images.T))
            mat = cp.einsum('kr,rcij->kcji', expkL, mat)
            if comp == 1:
                mat = mat[:, 0]
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out


def evaluate_density_wrapper(pairs_info, dm_slice, ignore_imag=True):
    c_driver = libgpbc.new_evaluate_density_driver_with_local_sort
    density_matrix_with_translation = cp.einsum("kt, ikpq->itpq", pairs_info["phase_diff_among_images"], dm_slice)
    n_channels, n_images, n_i_functions, n_j_functions = density_matrix_with_translation.shape

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
                 ctypes.cast(gaussians_per_angular_pair["i_shells"].data.ptr, ctypes.c_void_p),
                 ctypes.c_int(len(gaussians_per_angular_pair["i_shells"])),
                 ctypes.cast(gaussians_per_angular_pair["j_shells"].data.ptr, ctypes.c_void_p),
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

def sort_gaussian_pairs(mydf, xc_type="LDA", blocking_sizes=np.array([4, 4, 4])):
    blocking_sizes_on_gpu = cp.asarray(blocking_sizes)
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    lattice_vectors = np.asarray(cell.lattice_vectors())
    reciprocal_lattice_vectors = cp.asarray(np.linalg.inv(lattice_vectors.T))
    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        for dense, _ in tasks:
            pass
            # dense.mesh = tuple(np.asarray(np.ceil(np.array(dense.mesh) / blocking_sizes) * blocking_sizes, dtype=int))

        mydf.tasks = tasks

    pairs = []

    for grids_dense, grids_sparse in tasks:
        subcell_in_dense_region = grids_dense.cell

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.

            pairs.append({"mesh": grids_dense.mesh})

        else:
            # mesh = np.asarray(np.ceil(np.array(grids_dense.mesh) / blocking_sizes) * blocking_sizes, dtype=int)
            mesh = np.asarray(grids_dense.mesh)
            granularized_mesh_size = cp.ceil(cp.array(mesh) / blocking_sizes_on_gpu)
            granularized_mesh = cp.asarray(
                cp.meshgrid(*list(map(cp.arange, granularized_mesh_size)))).reshape(3, -1).transpose()

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
            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
            vectors_to_neighboring_images = gto.eval_gto.get_lattice_Ls(subcell_in_dense_region)
            phase_diff_among_images = cp.exp(
                1j * cp.asarray(mydf.kpts.reshape(-1, 3).dot(vectors_to_neighboring_images.T)))
            shell_to_ao_indices = gto.moleintor.make_loc(grouped_cell._bas, 'cart')
            per_angular_pairs = []
            for i_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF]):
                i_shells = np.where(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF] == i_angular)[0]
                n_i_shells = len(i_shells)
                i_basis = grouped_cell._bas[i_shells]

                i_exponents = cp.asarray(grouped_cell._env[i_basis[:, multigrid.PTR_EXP]])
                i_coord_pointers = np.asarray(grouped_cell._atm[i_basis[:, multigrid.ATOM_OF], PTR_COORD])
                i_x = cp.asarray(grouped_cell._env[i_coord_pointers])
                i_y = cp.asarray(grouped_cell._env[i_coord_pointers + 1])
                i_z = cp.asarray(grouped_cell._env[i_coord_pointers + 2])

                for j_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF]):
                    j_shells = np.where(
                        grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF] == j_angular)[0]
                    n_j_shells = len(j_shells)
                    j_basis = grouped_cell._bas[j_shells]

                    j_exponents = cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_EXP]])
                    j_coord_pointers = np.array(grouped_cell._atm[j_basis[:, multigrid.ATOM_OF], PTR_COORD])
                    j_x = cp.asarray(grouped_cell._env[j_coord_pointers])
                    j_y = cp.asarray(grouped_cell._env[j_coord_pointers + 1])
                    j_z = cp.asarray(grouped_cell._env[j_coord_pointers + 2])
                    pair_exponents = cp.repeat(j_exponents, i_exponents.size) + cp.tile(i_exponents, j_exponents.size)
                    real_space_cutoff_for_pairs = cp.sqrt(EIJ_CUTOFF / pair_exponents)

                    non_trivial_pairs_from_images = []
                    image_indices = []
                    mesh_begin_indices_from_images = []
                    mesh_size_from_images = []
                    for image_index, i_image in enumerate(vectors_to_neighboring_images):
                        shifted_i_x = i_x - i_image[0]
                        shifted_i_y = i_y - i_image[1]
                        shifted_i_z = i_z - i_image[2]

                        interatomic_distance = cp.square(cp.repeat(j_x, n_i_shells) - cp.tile(shifted_i_x, n_j_shells))
                        interatomic_distance += cp.square(cp.repeat(j_y, n_i_shells) - cp.tile(shifted_i_y, n_j_shells))
                        interatomic_distance += cp.square(cp.repeat(j_z, n_i_shells) - cp.tile(shifted_i_z, n_j_shells))
                        non_trivial_pairs = cp.where((cp.repeat(j_exponents, n_i_shells)
                                                      + cp.tile(i_exponents, n_j_shells)) /
                                                     pair_exponents * interatomic_distance < EIJ_CUTOFF)[0]

                        if len(non_trivial_pairs) == 0:
                            continue

                        pair_x = (cp.repeat(j_exponents * j_x, n_i_shells)
                                  + cp.tile(i_exponents * shifted_i_x, n_j_shells)) / pair_exponents
                        pair_x = pair_x[non_trivial_pairs]
                        pair_y = (cp.repeat(j_exponents * j_y, n_i_shells)
                                  + cp.tile(i_exponents * shifted_i_y, n_j_shells)) / pair_exponents
                        pair_y = pair_y[non_trivial_pairs]
                        pair_z = (cp.repeat(j_exponents * j_z, n_i_shells)
                                  + cp.tile(i_exponents * shifted_i_z, n_j_shells)) / pair_exponents
                        pair_z = pair_z[non_trivial_pairs]

                        real_space_cutoff_for_non_trivial_pairs = real_space_cutoff_for_pairs[non_trivial_pairs]
                        coordinates = cp.vstack((pair_x, pair_y, pair_z))
                        mesh_begin = cp.floor(reciprocal_lattice_vectors.dot(
                            coordinates - real_space_cutoff_for_non_trivial_pairs).T * cp.asarray(mesh))
                        mesh_end = cp.ceil(reciprocal_lattice_vectors.dot(
                            coordinates + real_space_cutoff_for_non_trivial_pairs).T * cp.asarray(mesh))
                        ranges = mesh_end - mesh_begin
                        broad_enough_pairs = cp.where(cp.all(ranges > 0, axis=1))[0]
                        if len(broad_enough_pairs) == 0:
                            continue

                        non_trivial_pairs_from_images.append(non_trivial_pairs[broad_enough_pairs])
                        image_indices.append(cp.repeat(cp.array(image_index, dtype=cp.int32), len(broad_enough_pairs)))
                        mesh_begin = mesh_begin[broad_enough_pairs]
                        mesh_end = mesh_end[broad_enough_pairs]
                        mesh_begin_indices_from_images.append(mesh_begin)
                        mesh_size_from_images.append(mesh_end - mesh_begin)

                    non_trivial_pairs_from_images = cp.asarray(cp.concatenate(non_trivial_pairs_from_images),
                                                               dtype=cp.int32)
                    image_indices = cp.asarray(cp.concatenate(image_indices), dtype=cp.int32)
                    mesh_begin_indices_from_images = cp.asarray(cp.concatenate(mesh_begin_indices_from_images),
                                                                dtype=cp.int32)
                    mesh_size_from_images = cp.asarray(cp.concatenate(mesh_size_from_images), dtype=cp.int32)

                    non_trivial_pairs_at_local_points = []
                    for granularized_mesh_point in granularized_mesh:
                        translation_lower = cp.floor(
                            (mesh_begin_indices_from_images - granularized_mesh_point) / cp.asarray(mesh))
                        translation_upper = cp.ceil(
                            (mesh_begin_indices_from_images
                             + mesh_size_from_images - granularized_mesh_point) / cp.asarray(mesh))
                        is_within_cell_translation = cp.all(translation_lower < translation_upper, axis=1)
                        non_trivial_pairs_at_local_points.append(cp.where(is_within_cell_translation)[0])

                    n_pairs_per_point = list(map(len, non_trivial_pairs_at_local_points))
                    accumulated_n_pairs_per_point = cp.asarray(accumulate(n_pairs_per_point), dtype=cp.int32)
                    non_trivial_pairs_at_local_points = cp.asarray(
                        cp.concatenate(non_trivial_pairs_at_local_points), dtype=cp.int32)
                    non_trivial_pairs_at_local_points = cp.asarray(non_trivial_pairs_at_local_points, dtype=cp.int32)

                    per_angular_pairs.append({
                        "angular": (i_angular, j_angular),
                        "non_trivial_pairs": non_trivial_pairs_from_images,
                        "non_trivial_pairs_at_local_points": non_trivial_pairs_at_local_points,
                        "accumulated_n_pairs_per_point": accumulated_n_pairs_per_point,
                        "image_indices": image_indices,
                        "i_shells": cp.asarray(i_shells, dtype=cp.int32),
                        "j_shells": cp.asarray(j_shells, dtype=cp.int32),
                        "shell_to_ao_indices": cp.asarray(shell_to_ao_indices, dtype=cp.int32),
                        "begin_indices": mesh_begin_indices_from_images,
                        "range": mesh_size_from_images
                    })

            pairs.append({
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": cp.asarray(vectors_to_neighboring_images),
                "phase_diff_among_images": phase_diff_among_images,
                "grouped_cell": grouped_cell,
                "mesh": mesh,
                "ao_indices_in_dense": cp.asarray(grids_dense.ao_idx),
                "ao_indices_in_sparse": cp.asarray(grids_sparse.ao_idx),
                "concatenated_ao_indices": cp.concatenate(
                    (cp.asarray(grids_dense.ao_idx), cp.asarray(grids_sparse.ao_idx))),
                "coeff_in_dense": cp.asarray(coeff_in_dense),
                "concatenated_coeff": cp.asarray(concatenated_coeff),
                "atm": cp.asarray(grouped_cell._atm, dtype=cp.int32),
                "bas": cp.asarray(grouped_cell._bas, dtype=cp.int32),
                "env": cp.asarray(grouped_cell._env),
                "lattice_vectors": cp.asarray(lattice_vectors),
                "reciprocal_lattice_vectors": reciprocal_lattice_vectors,
                "blocking_sizes": blocking_sizes
            })

    mydf.sorted_gaussian_pairs = pairs


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
        n_grid_points = np.prod(mesh)

        density = None
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
                coeff_sandwiched_density_matrix = cp.einsum('nkij,pi,qj->nkpq',
                                                            density_matrix_with_rows_in_dense,
                                                            pairs["coeff_in_dense"], pairs["concatenated_coeff"])

                density = evaluate_density_wrapper(pairs, coeff_sandwiched_density_matrix)
            else:
                raise NotImplementedError

        fft_grids = list(map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))

        weight_per_grid_point = 1. / n_k_points * mydf.cell.vol / n_grid_points
        density_contribution_on_g_mesh = tools.fft(density.reshape(n_channels * density_slices, -1),
                                                   mesh) * weight_per_grid_point
        new_intermediates.append(density_contribution_on_g_mesh.flatten())
        density_on_g_mesh[
            cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)] += density_contribution_on_g_mesh.reshape(
            (-1,) + tuple(mesh))

    density_on_g_mesh = density_on_g_mesh.reshape(n_channels, density_slices, -1)
    return density_on_g_mesh

def convert_veff_on_g_mesh_to_matrix(mydf, xc_for_fock_on_g_mesh, hermi=1, kpts=np.zeros((1, 3)), verbose=None):
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    xc_for_fock_on_g_mesh = xc_for_fock_on_g_mesh.reshape(-1, *mydf.mesh)
    nset = xc_for_fock_on_g_mesh.shape[0]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    at_gamma_point = multigrid.gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = cp.zeros((nset, nkpts, nao, nao))
    else:
        vj_kpts = cp.zeros((nset, nkpts, nao, nao), dtype=cp.complex128)

    skip = True
    for grids_dense, grids_sparse in tasks:

        mesh = grids_dense.mesh
        ngrids = cp.prod(mesh)
        log.debug('mesh %s', mesh)

        fft_grids = map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh)
        #:sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids)
        interpolated_xc_for_fock_on_gmesh = xc_for_fock_on_g_mesh[
            cp.ix_(cp.arange(xc_for_fock_on_g_mesh.shape[0]), *fft_grids)].reshape(nset, ngrids)

        reordered_veff_on_real_mesh = tools.ifft(interpolated_xc_for_fock_on_gmesh, mesh).reshape(nset, ngrids)
        veff_real_part = cp.asarray(reordered_veff_on_real_mesh.real, order='C')
        veff_imag_part = cp.asarray(reordered_veff_on_real_mesh.imag, order='C')
        ignore_vG_imag = hermi == 1 or abs(veff_imag_part.sum()) < multigrid.IMAG_TOL
        if ignore_vG_imag:
            reordered_veff_on_real_mesh = veff_real_part
        elif vj_kpts.dtype == cp.double:
            # ensure result complex array if tddft amplitudes are complex while
            # at gamma point
            vj_kpts = vj_kpts.astype(cp.complex128)

        ao_index_in_dense = grids_dense.ao_idx
        if grids_sparse is None:
            # ao_on_sliced_grid_in_dense, grid_begin, grid_end
            for ao_on_sliced_grid_in_dense, p0, p1 in mydf.aoR_loop(grids_dense, kpts):
                ao_values = ao_on_sliced_grid_in_dense[0]
                for k in range(nkpts):
                    for i in range(nset):
                        veff_scaled_ao = numint._scale_ao(ao_values[k], reordered_veff_on_real_mesh[i, p0:p1])
                        veff_sub_block = cp.dot(ao_values[k].conj().T, veff_scaled_ao)
                        vj_kpts[i, k, ao_index_in_dense[:, None], ao_index_in_dense] += veff_sub_block
                ao_values = ao_on_sliced_grid_in_dense = None
        else:
            ao_index_in_dense = grids_dense.ao_idx
            ao_index_in_sparse = grids_sparse.ao_idx
            # idx_t = numpy.append(idx_h, idx_l)
            n_ao_in_sparse = len(ao_index_in_dense)

            subcell_in_dense = grids_dense.cell
            subcell_in_sparse = grids_sparse.cell
            decontracted_subcell_in_dense, decontracted_coeff_in_dense = subcell_in_dense.decontract_basis(
                to_cart=True, aggregate=True)
            decontracted_subcell_in_sparse, decontracted_coeff_in_sparse = subcell_in_sparse.decontract_basis(
                to_cart=True, aggregate=True)
            concatenated_cell = decontracted_subcell_in_dense + decontracted_subcell_in_sparse
            concatenated_coeff = scipy.linalg.block_diag(decontracted_coeff_in_dense, decontracted_coeff_in_sparse)

            n_shells_in_dense = multigrid._pgto_shells(subcell_in_dense)
            n_shells_in_total = multigrid._pgto_shells(concatenated_cell)
            shells_indices = (0, n_shells_in_dense, 0, n_shells_in_total)
            veff_slice = eval_mat(concatenated_cell, veff_real_part, shells_indices, 1, 'LDA', kpts)
            # Imaginary part may contribute
            if not ignore_vG_imag:
                veff_slice += eval_mat(concatenated_cell, veff_imag_part, shells_indices, 1, 'LDA', kpts) * 1j

            veff_slice = cp.einsum('nkpq,pi,qj->nkij', cp.asarray(veff_slice),
                                   cp.asarray(decontracted_coeff_in_dense), cp.asarray(concatenated_coeff))

            vj_kpts[:, :, ao_index_in_dense[:, None], ao_index_in_dense] += veff_slice[:, :, :, :n_ao_in_sparse]
            vj_kpts[:, :, ao_index_in_dense[:, None], ao_index_in_sparse] += veff_slice[:, :, :, n_ao_in_sparse:]

            if hermi == 1:
                vj_kpts[:, :, ao_index_in_sparse[:, None], ao_index_in_dense] += \
                    veff_slice[:, :, :, n_ao_in_sparse:].transpose(0, 1, 3, 2).conj()
            else:
                shells_indices = (n_shells_in_dense, n_shells_in_total, 0, n_shells_in_dense)
                veff_slice = eval_mat(concatenated_cell, veff_real_part, shells_indices, 1, 'LDA', kpts)
                # Imaginary part may contribute
                if not ignore_vG_imag:
                    veff_slice += eval_mat(concatenated_cell, veff_imag_part, shells_indices, 1, 'LDA', kpts) * 1j
                veff_slice = cp.einsum('nkpq,pi,qj->nkij', veff_slice, decontracted_coeff_in_sparse,
                                       decontracted_coeff_in_dense)
                vj_kpts[:, :, ao_index_in_sparse[:, None], ao_index_in_dense] += veff_slice

    return vj_kpts


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
        xc_for_fock = convert_veff_on_g_mesh_to_matrix(mydf, weighted_xc_for_fock_on_g_mesh, hermi, kpts_band,
                                                       verbose=log)
    elif xc_type == 'GGA':
        raise NotImplementedError

    if return_j:
        vj = convert_veff_on_g_mesh_to_matrix(mydf, coulomb_on_g_mesh, hermi, kpts_band, verbose=log)
        vj = fft_jk._format_jks(xc_for_fock, dm_kpts, input_band, kpts)
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
