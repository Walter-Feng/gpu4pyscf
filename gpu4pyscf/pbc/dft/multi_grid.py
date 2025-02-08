import pyscf.pbc.gto as gto
import gpu4pyscf.pbc.df.fft as fft
import gpu4pyscf.pbc.df.fft_jk as fft_jk
from pyscf.dft.numint import libdft

import gpu4pyscf.pbc.scf.hf
from gpu4pyscf.lib import logger
from gpu4pyscf.pbc.dft import numint
from gpu4pyscf.pbc import tools
import gpu4pyscf.lib.cupy_helper as cupy_helper

from pyscf.pbc.dft.multigrid import multigrid
from pyscf.pbc.df.df_jk import _format_kpts_band

import numpy as np
import sys
import cupy as cp
import scipy

import ctypes

libgdft = cupy_helper.load_library('libgdft')
libgpbc = cupy_helper.load_library('libgpbc')

PTR_COORD = 1
EIJ_CUTOFF = 60


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


def evaluate_density(cell, dm, shells_slice=None, xc_type='LDA', kpts=None,
                     global_mesh=None, offset=None, local_mesh=None, ignore_imag=False,
                     out=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Kwargs:
        ignore_image :
            The output density is assumed to be real if ignore_imag=True.
    '''
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if global_mesh is None:
        global_mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(global_mesh) / vol
    minimum_exponent = np.hstack(cell.bas_exps()).min()
    theta_ij = minimum_exponent / 2
    lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_summation_factor
    if xc_type != 'LDA':
        precision *= .1
    # concatenate two molecules
    print("before concatenation:", cell._bas)
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    exp_drop_factor = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)
    env[multigrid.PTR_EXPDROP] = exp_drop_factor
    print("after concatenation:", bas)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shells_slice is None:
        shells_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shells_slice
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    dm = cp.asarray(dm, order='C')
    assert (dm.shape[-2:] == (naoi, naoj))

    vectors_to_neighboring_images = gto.eval_gto.get_lattice_Ls(cell)

    if cell.dimension == 0 or kpts is None or multigrid.gamma_point(kpts):
        n_k_points, n_images = 1, vectors_to_neighboring_images.shape[0]
        dm = dm.reshape(-1, 1, naoi, naoj).transpose(0, 1, 3, 2)
    else:
        phase_diff_among_images = np.exp(1j * kpts.reshape(-1, 3).dot(vectors_to_neighboring_images.T))
        n_k_points, n_images = phase_diff_among_images.shape
        dm = dm.reshape(-1, n_k_points, naoi, naoj).transpose(0, 1, 3, 2)
    n_dm = dm.shape[0]

    lattice_vector = cell.lattice_vectors()
    reciprocal_lattice_vector = np.linalg.inv(lattice_vector.T)
    if offset is None:
        offset = (0, 0, 0)
    if local_mesh is None:
        local_mesh = global_mesh
    precision_in_log = np.log(precision * multigrid.EXTRA_PREC)

    if abs(lattice_vector - np.diag(lattice_vector.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    xc_type = xc_type.upper()
    if xc_type == 'LDA':
        n_components = 1
    elif xc_type == 'GGA':
        n_components = 4
    else:
        raise NotImplementedError('meta-GGA')
    if n_components == 1:
        density_shape = (np.prod(local_mesh),)
    else:
        density_shape = (n_components, np.prod(local_mesh))
    kernel_name = 'NUMINTrho_' + xc_type.lower() + lattice_type
    driver = libdft.NUMINT_rho_drv

    new_driver = libgpbc.evaluate_density_driver

    def driver_wrapper(density_shape, dm):
        density = np.zeros(density_shape, order='C')
        driver(getattr(libdft, kernel_name),
               density.ctypes.data_as(ctypes.c_void_p),
               dm.get().ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(n_components), ctypes.c_int(0),
               (ctypes.c_int * 4)(i0, i1, j0, j1),
               ao_loc.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_double(precision_in_log),
               ctypes.c_int(cell.dimension),
               ctypes.c_int(n_images),
               vectors_to_neighboring_images.ctypes.data_as(ctypes.c_void_p),
               lattice_vector.ctypes.data_as(ctypes.c_void_p),
               reciprocal_lattice_vector.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*local_mesh),
               (ctypes.c_int * 3)(*global_mesh),
               atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
               bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
               env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(density)

    atm_on_gpu = cp.asarray(cell._atm)
    bas_on_gpu = cp.asarray(cell._bas)
    env_on_gpu = cp.asarray(cell._env)
    ao_loc_on_gpu = cp.asarray(ao_loc)
    vectors_to_neighboring_images_on_gpu = cp.asarray(vectors_to_neighboring_images)

    def new_driver_wrapper(density, dm):
        assert isinstance(density, cp.ndarray)
        new_driver(ctypes.cast(density.data.ptr, ctypes.c_void_p),
                   ctypes.cast(dm.data.ptr, ctypes.c_void_p),
                   (ctypes.c_int * 4)(i0, i1, j0 - cell.nbas, j1 - cell.nbas),
                   ctypes.cast(ao_loc_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.c_int(0), ctypes.c_int(0),
                   ctypes.c_int(n_images),
                   ctypes.cast(vectors_to_neighboring_images_on_gpu.data.ptr, ctypes.c_void_p),
                   lattice_vector.ctypes.data_as(ctypes.c_void_p),
                   (ctypes.c_int * 3)(*offset),
                   (ctypes.c_int * 3)(*local_mesh),
                   (ctypes.c_int * 3)(*global_mesh),
                   ctypes.cast(atm_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(bas_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(env_on_gpu.data.ptr, ctypes.c_void_p))

        return density

    def another_new_driver_wrapper(density, dm):

        another_new_driver = libgpbc.new_evaluate_density_driver

        lattice_vector_on_gpu = cp.asarray(lattice_vector)
        reciprocal_lattice_vector_on_gpu = cp.asarray(reciprocal_lattice_vector)
        dm_copy = dm.transpose(0, 2, 1)

        assert (dm_copy.shape[1:] == (naoi, naoj))
        print(shells_slice)

        for i_angular in set(cell._bas[shells_slice[0]:shells_slice[1], multigrid.ANG_OF]):
            i_shells = np.where(cell._bas[shells_slice[0]:shells_slice[1], multigrid.ANG_OF] == i_angular)[0] + \
                       shells_slice[0]
            n_i_shells = len(i_shells)
            i_basis = cell._bas[i_shells]
            n_functions_per_i_shell = (i_angular + 1) * (i_angular + 2) // 2
            i_functions = cp.repeat(
                cp.asarray(ao_loc[i_shells]), n_functions_per_i_shell).reshape(
                -1, n_functions_per_i_shell) + cp.arange(n_functions_per_i_shell)

            i_exponents = cp.asarray(cell._env[i_basis[:, multigrid.PTR_EXP]])
            i_coord_pointers = np.array(cell._atm[i_basis[:, multigrid.ATOM_OF], PTR_COORD])
            i_x = cp.asarray(cell._env[i_coord_pointers])
            i_y = cp.asarray(cell._env[i_coord_pointers + 1])
            i_z = cp.asarray(cell._env[i_coord_pointers + 2])
            i_coeff = cp.asarray(cell._env[i_basis[:, multigrid.PTR_COEFF]])

            for j_angular in set(cell._bas[shells_slice[2]:shells_slice[3], multigrid.ANG_OF]):
                j_shells = shells_slice[2] + \
                           np.where(cell._bas[shells_slice[2]:shells_slice[3], multigrid.ANG_OF] == j_angular)[0]
                n_j_shells = len(j_shells)
                j_basis = cell._bas[j_shells]
                n_functions_per_j_shell = (j_angular + 1) * (j_angular + 2) // 2

                j_exponents = cp.asarray(cell._env[j_basis[:, multigrid.PTR_EXP]])
                j_coord_pointers = np.array(cell._atm[j_basis[:, multigrid.ATOM_OF], PTR_COORD])
                j_x = cp.asarray(cell._env[j_coord_pointers])
                j_y = cp.asarray(cell._env[j_coord_pointers + 1])
                j_z = cp.asarray(cell._env[j_coord_pointers + 2])
                j_coeff = cp.asarray(cell._env[j_basis[:, multigrid.PTR_COEFF]])
                pair_exponents = cp.repeat(j_exponents, n_i_shells) + cp.tile(i_exponents, n_j_shells)
                real_space_cutoff_for_pairs = cp.sqrt(EIJ_CUTOFF / pair_exponents)
                pair_coeff = cp.repeat(j_coeff, i_coeff.size) * cp.tile(i_coeff, j_coeff.size)

                non_trivial_pairs_from_images = []
                image_indices = []
                mesh_begin_indices_from_images = []
                mesh_end_indices_from_images = []

                for image_index, i_image in enumerate(vectors_to_neighboring_images):
                    shifted_i_x = i_x - i_image[0]
                    shifted_i_y = i_y - i_image[1]
                    shifted_i_z = i_z - i_image[2]

                    interatomic_distance = cp.square(cp.repeat(j_x, n_i_shells) - cp.tile(shifted_i_x, n_j_shells))
                    interatomic_distance += cp.square(cp.repeat(j_y, n_i_shells) - cp.tile(shifted_i_y, n_j_shells))
                    interatomic_distance += cp.square(cp.repeat(j_z, n_i_shells) - cp.tile(shifted_i_z, n_j_shells))
                    prefactor = pair_coeff * cp.exp(- (cp.repeat(j_exponents, n_i_shells)
                                                       + cp.tile(i_exponents, n_j_shells)) /
                                                    pair_exponents * interatomic_distance)

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
                    mesh_begin = cp.floor(reciprocal_lattice_vector_on_gpu.dot(
                        coordinates - real_space_cutoff_for_non_trivial_pairs).T * cp.asarray(global_mesh))
                    mesh_end = cp.ceil(reciprocal_lattice_vector_on_gpu.dot(
                        coordinates + real_space_cutoff_for_non_trivial_pairs).T * cp.asarray(global_mesh))
                    ranges = mesh_end - mesh_begin
                    broad_enough_pairs = cp.where(cp.all(ranges > 0, axis=1))[0]
                    if len(broad_enough_pairs) == 0:
                        continue

                    non_trivial_pairs_from_images.append(non_trivial_pairs[broad_enough_pairs])
                    image_indices.append(cp.repeat(cp.array(image_index,dtype=cp.int32), len(broad_enough_pairs)))
                    mesh_begin_indices_from_images.append(cp.floor(mesh_begin[broad_enough_pairs] / 4) * 4)
                    mesh_end_indices_from_images.append(cp.ceil(mesh_end[broad_enough_pairs] / 4) * 4)

                non_trivial_pairs_from_images = cp.asarray(cp.concatenate(non_trivial_pairs_from_images), dtype=cp.int32)
                image_indices = cp.asarray(cp.concatenate(image_indices), dtype=cp.int32)

                print("non_trivial_pairs: ", non_trivial_pairs_from_images)
                print("image_indices: ", image_indices)
                print("global mesh: ", global_mesh)

                i_shells_on_gpu = cp.asarray(i_shells, dtype=cp.int32)
                j_shells_on_gpu = cp.asarray(j_shells, dtype=cp.int32)

                another_new_driver(ctypes.cast(density.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(dm_copy.data.ptr, ctypes.c_void_p),
                                   ctypes.c_int(i_angular), ctypes.c_int(j_angular),
                                   ctypes.cast(non_trivial_pairs_from_images.data.ptr, ctypes.c_void_p),
                                   ctypes.c_int(len(non_trivial_pairs_from_images)),
                                   ctypes.cast(i_shells_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.c_int(len(i_shells)),
                                   ctypes.cast(j_shells_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.c_int(len(j_shells)),
                                   ctypes.cast(ao_loc_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.c_int(naoj),
                                   ctypes.cast(image_indices.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(vectors_to_neighboring_images_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(lattice_vector_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(reciprocal_lattice_vector_on_gpu.data.ptr, ctypes.c_void_p),
                                   (ctypes.c_int * 3)(*global_mesh),
                                   ctypes.cast(atm_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(bas_on_gpu.data.ptr, ctypes.c_void_p),
                                   ctypes.cast(env_on_gpu.data.ptr, ctypes.c_void_p))

                return density

    rho = []
    for i, dm_i in enumerate(dm):
        if cell.dimension == 0:
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == cp.complex128
            if has_imag:
                dmR = cp.asarray(dm_i.real, order='C')
                dmI = cp.asarray(dm_i.imag, order='C')
            else:
                # make a copy because the dm may be overwritten in the
                # NUMINT_rho_drv inplace
                dmR = cp.array(dm_i, order='C', copy=True)

        elif kpts is None or multigrid.gamma_point(kpts):
            if ignore_imag:
                # basis are real. dm.imag can be dropped if ignore_imag
                dm_i = dm_i.real
            has_imag = dm_i.dtype == cp.complex128
            if has_imag:
                dmR = cp.repeat(dm_i.real, n_images, axis=0)
                dmI = cp.repeat(dm_i.imag, n_images, axis=0)
            else:
                dmR = cp.repeat(dm_i, n_images, axis=0)

        else:
            dm_L = cp.dot(phase_diff_among_images.T, dm_i.reshape(n_k_points, -1)).reshape(n_images, naoj, naoi)
            dmR = cp.asarray(dm_L.real, order='C')

            if ignore_imag:
                has_imag = False
            else:
                dmI = cp.asarray(dm_L.imag, order='C')
                has_imag = (abs(dmI).max() > 1e-8)
                if (has_imag and xc_type == 'LDA' and
                        naoi == naoj and
                        # For hermitian density matrices, the anti-symmetry
                        # character of the imaginary part of the density matrices
                        # can be found by rearranging the repeated images.
                        abs(dm_i - dm_i.conj().transpose(0, 2, 1)).max() < 1e-8):
                    has_imag = False
            dm_L = None
        if has_imag:
            # complex density cannot be updated inplace directly by
            # function NUMINT_rho_drv
            if out is None:
                rho_i = cp.empty(density_shape, cp.complex128)
                new_driver_wrapper(rho_i.real, dmR)
                new_driver_wrapper(rho_i.imag, dmI)
            else:
                assert out[i].dtype == cp.complex128
                new_driver_wrapper(out[i].real, dmR)
                new_driver_wrapper(out[i].imag, dmI)
        else:
            pass
            # if out is None:
            #     # rho_i needs to be initialized to 0 because rho_i is updated
            #     # inplace in function NUMINT_rho_drv
            #     # rho_i = driver_wrapper(shape, dmR, hermi)
            #     rho_i = new_driver_wrapper(cp.zeros(density_shape), dmR)
            # else:
            #     assert out[i].dtype == cp.double
            #     rho_i = new_driver_wrapper(out[i], dmR)

        dmR = dmR * 0 + 1
        cpu_driver = driver_wrapper(density_shape, dmR)
        print("from old")
        from_old_driver = new_driver_wrapper(cp.zeros(density_shape), dmR)
        print("from new")
        from_new_driver = another_new_driver_wrapper(cp.zeros(density_shape), dmR)
        print(shells_slice)
        print(from_new_driver)
        print(cpu_driver)
        print(from_new_driver / from_old_driver)

        assert 0
        dmR = dmI = None
        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]

    return cp.asarray(rho)


def _eval_rho_bra(cell, dms, shell_ranges, xc_type, kpts, grids, ignore_imag, log):
    lattice_vectors = np.asarray(cell.lattice_vectors())
    max_element = lattice_vectors.max()
    global_mesh = np.asarray(grids.mesh)
    real_space_cutoff = grids.cell.rcut
    nset = dms.shape[0]
    if xc_type == 'LDA':
        rho_slices = 1
    else:
        rho_slices = 4

    if real_space_cutoff > max_element * multigrid.R_RATIO_SUBLOOP:
        density = evaluate_density(cell, dms, shell_ranges, xc_type, kpts, global_mesh, ignore_imag=ignore_imag)
        return cp.reshape(density, (nset, rho_slices, np.prod(global_mesh)))

    if ignore_imag:
        density = cp.zeros((nset, rho_slices) + tuple(global_mesh))
    else:
        density = cp.zeros((nset, rho_slices) + tuple(global_mesh), dtype=cp.complex128)

    b = np.linalg.inv(lattice_vectors.T)
    row_shell_begin, row_shell_end, col_shell_begin, col_shell_end = shell_ranges
    n_col_shells = col_shell_end - col_shell_begin
    copied_cell = cell.copy(deep=False)
    rest_dms = []
    rest_bas = []
    i1 = 0
    for atom in set(cell._bas[row_shell_begin:row_shell_end, multigrid.ATOM_OF]):
        row_shells = np.where(cell._bas[row_shell_begin:row_shell_end, multigrid.ATOM_OF] == atom)[0]
        row_basis = cell._bas[row_shells]
        l = row_basis[:, multigrid.ANG_OF]
        i0, i1 = i1, i1 + sum((l + 1) * (l + 2) // 2)
        density_matrix_subblock = dms[:, :, i0:i1]

        atom_position = cell.atom_coord(atom)
        local_mesh_begin_fractional = b.dot(atom_position - real_space_cutoff)
        local_mesh_end_fractional = b.dot(atom_position + real_space_cutoff)

        if (np.all(0 < local_mesh_begin_fractional) and np.all(local_mesh_end_fractional < 1)):
            copied_cell._bas = np.vstack((row_basis, cell._bas[col_shell_begin:col_shell_end]))
            n_row_shells = len(row_shells)
            sub_slice = (0, n_row_shells, n_row_shells, n_row_shells + n_col_shells)

            head = (local_mesh_begin_fractional * global_mesh).astype(int)
            tail = np.ceil(local_mesh_end_fractional * global_mesh).astype(int)
            local_mesh = tail - head
            log.debug1('atm %d  rcut %f  offset %s submesh %s',
                       atom, real_space_cutoff, head, local_mesh)
            evaluated_density = evaluate_density(copied_cell, density_matrix_subblock, sub_slice, xc_type, kpts,
                                                 global_mesh, head, local_mesh, ignore_imag=ignore_imag)
            #:rho[:,:,offset[0]:mesh1[0],offset[1]:mesh1[1],offset[2]:mesh1[2]] += \
            #:        numpy.reshape(rho1, (nset, rhodim) + tuple(submesh))
            gx = cp.arange(head[0], tail[0], dtype=cp.int32)
            gy = cp.arange(head[1], tail[1], dtype=cp.int32)
            gz = cp.arange(head[2], tail[2], dtype=cp.int32)

            density[cp.ix_(cp.arange(nset), cp.arange(rho_slices), gx, gy, gz)] += evaluated_density.reshape(
                -1, rho_slices, *local_mesh)
        else:
            log.debug1('atm %d  rcut %f  over 2 images', atom, real_space_cutoff)
            #:rho1 = eval_rho(pcell, sub_dms, sub_slice, hermi, xc_type, kpts,
            #:                mesh, ignore_imag=ignore_imag)
            #:rho += numpy.reshape(rho1, rho.shape)
            # or
            #:eval_rho(pcell, sub_dms, sub_slice, hermi, xc_type, kpts,
            #:         mesh, ignore_imag=ignore_imag, out=rho)
            rest_bas.append(row_basis)
            rest_dms.append(density_matrix_subblock)
    if rest_bas:
        copied_cell._bas = np.vstack(rest_bas + [cell._bas[col_shell_begin:col_shell_end]])
        n_row_shells = sum(len(x) for x in rest_bas)
        sub_slice = (0, n_row_shells, n_row_shells, n_row_shells + n_col_shells)
        density_matrix_subblock = cp.concatenate(rest_dms, axis=2)
        # Update density matrix in place
        evaluate_density(copied_cell, density_matrix_subblock, sub_slice, xc_type, kpts,
                         global_mesh, ignore_imag=ignore_imag, out=density)

    return density.reshape((nset, rho_slices, np.prod(global_mesh)))


def sort_gaussian_pairs(mydf, xc_type="LDA", blocking_size=4):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    lattice_vectors = np.asarray(cell.lattice_vectors())
    reciprocal_lattice_vectors = cp.asarray(np.linalg.inv(lattice_vectors.T))
    n_k_points = mydf.kpts.shape[0]
    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)

    pairs = []

    for grids_dense, grids_sparse in tasks:
        subcell_in_dense_region = grids_dense.cell

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            pairs.append(None)
        else:
            mesh = tuple(np.ceil(np.array(grids_dense.mesh) / blocking_size) * blocking_size)

            subcell_in_dense_region.mesh = mesh
            n_grid_points = np.prod(mesh)
            weight_per_grid_point = 1. / n_k_points * cell.vol / n_grid_points
            subcell_in_sparse_region = grids_sparse.cell
            equivalent_cell_in_dense, primitive_coeff_in_dense = subcell_in_dense_region.decontract_basis(to_cart=True,
                                                                                                          aggregate=True)
            equivalent_cell_in_sparse, primitive_coeff_in_sparse = subcell_in_sparse_region.decontract_basis(
                to_cart=True, aggregate=True)
            grouped_cell = equivalent_cell_in_dense + equivalent_cell_in_sparse

            vol = grouped_cell.vol
            weight_penalty = np.prod(grouped_cell.mesh) / vol
            minimum_exponent = np.hstack(grouped_cell.bas_exps()).min()
            theta_ij = minimum_exponent / 2
            lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
            precision = grouped_cell.precision / weight_penalty / lattice_summation_factor
            if xc_type != 'LDA':
                precision *= .1
            exp_drop_factor = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)

            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(grouped_cell)
            vectors_to_neighboring_images = gto.eval_gto.get_lattice_Ls(subcell_in_dense_region)
            shell_to_ao_indices = gto.moleintor.make_loc(grouped_cell._bas, 'cart')

            per_angular_pairs = []
            for i_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF]):
                i_shells = np.where(grouped_cell._bas[0:n_primitive_gtos_in_dense, multigrid.ANG_OF] == i_angular)[
                    0]
                i_basis = grouped_cell._bas[i_shells]
                n_functions_per_i_shell = (i_angular + 1) * (i_angular + 2) // 2
                i_functions = cp.repeat(
                    cp.asarray(shell_to_ao_indices[i_shells]), n_functions_per_i_shell).reshape(
                    -1, n_functions_per_i_shell) + cp.arange(n_functions_per_i_shell)
                i_functions = i_functions.flatten()

                i_exponents = cp.asarray(grouped_cell._env[i_basis[:, multigrid.PTR_EXP]])
                i_coord_pointers = np.asarray(grouped_cell._atm[i_basis[:, multigrid.ATOM_OF], PTR_COORD])
                i_x = cp.asarray(grouped_cell._env[i_coord_pointers])
                i_y = cp.asarray(grouped_cell._env[i_coord_pointers + 1])
                i_z = cp.asarray(grouped_cell._env[i_coord_pointers + 2])
                i_coeff = cp.asarray(grouped_cell._env[i_basis[:, multigrid.PTR_COEFF]])

                for j_angular in set(grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF]):
                    j_shells = \
                        np.where(
                            grouped_cell._bas[0:n_primitive_gtos_in_two_regions, multigrid.ANG_OF] == j_angular)[
                            0]
                    j_basis = grouped_cell._bas[j_shells]
                    n_functions_per_j_shell = (j_angular + 1) * (j_angular + 2) // 2
                    j_functions = cp.repeat(
                        cp.asarray(shell_to_ao_indices[j_shells]), n_functions_per_j_shell).reshape(
                        -1, n_functions_per_j_shell) + cp.arange(n_functions_per_j_shell)
                    j_functions = j_functions.flatten()

                    j_exponents = cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_EXP]])
                    j_coord_pointers = np.array(grouped_cell._atm[j_basis[:, multigrid.ATOM_OF], PTR_COORD])
                    j_x = cp.asarray(grouped_cell._env[j_coord_pointers])
                    j_y = cp.asarray(grouped_cell._env[j_coord_pointers + 1])
                    j_z = cp.asarray(grouped_cell._env[j_coord_pointers + 2])
                    j_coeff = cp.asarray(grouped_cell._env[j_basis[:, multigrid.PTR_COEFF]])
                    pair_exponents = cp.repeat(j_exponents, i_exponents.size) + cp.tile(i_exponents, j_exponents.size)
                    real_space_cutoff_for_pairs = cp.sqrt(EIJ_CUTOFF / pair_exponents)
                    pair_coeff = cp.repeat(j_coeff, i_coeff.size) * cp.tile(i_coeff, j_coeff.size)

                    non_trivial_pairs_from_images = []
                    image_indices = []
                    mesh_begin_indices_from_images = []
                    mesh_end_indices_from_images = []
                    for image_index, i_image in enumerate(vectors_to_neighboring_images):
                        shifted_j_x = j_x + i_image[0]
                        shifted_j_y = j_y + i_image[1]
                        shifted_j_z = j_z + i_image[2]

                        interatomic_distance = cp.square(
                            cp.repeat(shifted_j_x, i_exponents.size) - cp.tile(i_x, j_exponents.size))
                        interatomic_distance += cp.square(
                            cp.repeat(shifted_j_y, i_exponents.size) - cp.tile(i_y, j_exponents.size))
                        interatomic_distance += cp.square(
                            cp.repeat(shifted_j_z, i_exponents.size) - cp.tile(i_z, j_exponents.size))
                        prefactor = pair_coeff * cp.exp(- (cp.repeat(j_exponents, i_exponents.size)
                                                           + cp.tile(i_exponents, j_exponents.size)) /
                                                        pair_exponents * interatomic_distance)
                        non_trivial_pairs = cp.where(prefactor > exp_drop_factor)[0]

                        if len(non_trivial_pairs) == 0:
                            continue

                        pair_x = (cp.repeat(j_exponents * shifted_j_x, i_exponents.size)
                                  + cp.tile(i_exponents * i_x, j_exponents.size)) / pair_exponents
                        pair_x = pair_x[non_trivial_pairs]
                        pair_y = (cp.repeat(j_exponents * shifted_j_y, i_exponents.size)
                                  + cp.tile(i_exponents * i_y, j_exponents.size)) / pair_exponents
                        pair_y = pair_y[non_trivial_pairs]
                        pair_z = (cp.repeat(j_exponents * shifted_j_z, i_exponents.size)
                                  + cp.tile(i_exponents * i_z, j_exponents.size)) / pair_exponents
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
                        image_indices.append(cp.ones(len(broad_enough_pairs), dtype=cp.int32) * image_index)
                        mesh_begin_indices_from_images.append(cp.floor(mesh_begin[broad_enough_pairs] / 4) * 4)
                        mesh_end_indices_from_images.append(cp.ceil(mesh_end[broad_enough_pairs] / 4) * 4)

                    non_trivial_pairs_from_images = cp.concatenate(non_trivial_pairs_from_images)
                    image_indices = cp.concatenate(image_indices)

                    per_angular_pairs.append({
                        "angular": (i_angular, j_angular),
                        "non_trivial_pairs": non_trivial_pairs_from_images,
                        "image_indices": image_indices,
                        "i_shells": i_shells,
                        "i_functions": i_functions,
                        "j_shells": j_shells,
                        "j_functions": j_functions
                    })

                    # mesh_begin_indices_from_images = cp.concatenate(mesh_begin_indices_from_images)
                    # mesh_end_indices_from_images = cp.concatenate(mesh_end_indices_from_images)
                    # print(mesh_end_indices_from_images - mesh_begin_indices_from_images)
                    # print(mesh_begin_indices_from_images)
                    # print(mesh_end_indices_from_images)

            pairs.append({
                "per_angular_pairs": per_angular_pairs,
                "neighboring_images": vectors_to_neighboring_images,
                "grouped_cell": grouped_cell
            })

    mydf.sorted_gaussian_pairs = pairs


def new_evaluate_density_on_g_mesh(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0, rho_g_high_order=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell

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
        mesh = tuple(grids_dense.mesh)
        n_grid_points = np.prod(mesh)

        if grids_sparse is None:
            assert pairs is None

            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            density = cp.zeros((n_channels, density_slices, n_grid_points), dtype=cp.complex128)
            ao_indices_in_dense = grids_dense.ao_idx
            density_matrix_in_dense_region = cp.asarray(dms[:, :, ao_indices_in_dense[:, None], ao_indices_in_dense],
                                                        order='C')
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

            fft_grids = list(
                map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))
            #:rhoG[:,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape((-1,)+mesh)

            density_contribution_on_g_mesh = None
            density_on_g_mesh[
                cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)] += density_contribution_on_g_mesh.reshape(
                (-1,) + mesh)
            assert 0


def evaluate_density_on_g_mesh(mydf, dm_kpts, hermi=1, kpts=np.zeros((1, 3)), deriv=0, rho_g_high_order=None):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell

    dm_kpts = cp.asarray(dm_kpts, order='C')
    dms = fft_jk._format_dms(dm_kpts, kpts)
    n_channels, n_k_points, nao = dms.shape[:3]

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        mydf.tasks = tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        log.debug('Multigrid ntasks %s', len(tasks))

    assert (deriv < 1)
    # hermi = hermi and abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9
    gga_high_order = False
    density_slices = 1  # Presumably
    if deriv == 0:
        xc_type = 'LDA'

    elif deriv == 1:
        if rho_g_high_order is not None:
            raise NotImplementedError

    elif deriv == 2:  # meta-GGA
        raise NotImplementedError

    nx, ny, nz = mydf.mesh
    density_on_g_mesh = cp.zeros((n_channels * density_slices, nx, ny, nz), dtype=cp.complex128)
    for grids_dense, grids_sparse in tasks:
        subcell_in_dense_region = grids_dense.cell
        mesh = tuple(grids_dense.mesh)
        n_grid_points = np.prod(mesh)
        log.debug('mesh %s  rcut %g', mesh, subcell_in_dense_region.rcut)

        if grids_sparse is None:
            # The first pass handles all diffused functions using the regular
            # matrix multiplication code.
            density = cp.zeros((n_channels, density_slices, n_grid_points), dtype=cp.complex128)
            ao_indices_in_dense = grids_dense.ao_idx
            density_matrix_in_dense_region = cp.asarray(dms[:, :, ao_indices_in_dense[:, None], ao_indices_in_dense],
                                                        order='C')
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
            ao_indices_in_dense = grids_dense.ao_idx
            ao_indices_in_sparse = grids_sparse.ao_idx
            concatenated_ao_indices = cp.append(ao_indices_in_dense, ao_indices_in_sparse)

            density_matrix_block_from_dense_rows = cp.asarray(
                dms[:, :, ao_indices_in_dense[:, None], concatenated_ao_indices], order='C')
            density_matrix_block_from_sparse_rows = cp.asarray(
                dms[:, :, ao_indices_in_sparse[:, None], ao_indices_in_dense], order='C')

            # guessing density matrix that have both AOs from sparse region is neglected

            subcell_in_sparse_region = grids_sparse.cell
            equivalent_cell_in_dense, coeff_from_decontraction = subcell_in_dense_region.decontract_basis(to_cart=True,
                                                                                                          aggregate=True)
            equivalent_cell_in_sparse, primitive_coeff_in_sparse = subcell_in_sparse_region.decontract_basis(
                to_cart=True, aggregate=True)
            concatenated_cell = equivalent_cell_in_dense + equivalent_cell_in_sparse
            concatenated_coeff = scipy.linalg.block_diag(coeff_from_decontraction, primitive_coeff_in_sparse)

            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(concatenated_cell)

            if deriv == 0:
                if hermi:
                    n_ao_in_sparse, n_ao_in_dense = density_matrix_block_from_sparse_rows.shape[2:]
                    density_matrix_block_from_dense_rows[:, :, :,
                    n_ao_in_dense:] += density_matrix_block_from_sparse_rows.transpose(0, 1, 3, 2)
                    coeff_sandwiched_density_matrix = cp.einsum('nkij,pi,qj->nkpq',
                                                                density_matrix_block_from_dense_rows,
                                                                coeff_from_decontraction, concatenated_coeff)
                    shells_slice = (0, n_primitive_gtos_in_dense, 0, n_primitive_gtos_in_two_regions)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None, ignore_imag=True)
                    density = _eval_rho_bra(concatenated_cell, coeff_sandwiched_density_matrix, shells_slice,
                                            'LDA', kpts, grids_dense, True, log)
                else:
                    raise NotImplementedError

            elif deriv == 1:
                raise NotImplementedError

        weight_per_grid_point = 1. / n_k_points * cell.vol / n_grid_points
        density_contribution_on_g_mesh = tools.fft(density.reshape(n_channels * density_slices, -1), mesh)
        density_contribution_on_g_mesh *= weight_per_grid_point
        fft_grids = list(map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))
        #:rhoG[:,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape((-1,)+mesh)

        density_on_g_mesh[
            cp.ix_(cp.arange(n_channels * density_slices), *fft_grids)] += density_contribution_on_g_mesh.reshape(
            (-1,) + mesh)

    density_on_g_mesh = density_on_g_mesh.reshape(n_channels, density_slices, -1)

    if gga_high_order:
        g_vectors = cell.get_Gv(mydf.mesh)
        vector_density_on_g_mesh = cp.einsum('np,px->nxp', 1j * density_on_g_mesh[:, 0], g_vectors)
        density_on_g_mesh = cp.concatenate([density_on_g_mesh, vector_density_on_g_mesh], axis=1)
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
    '''Compute the XC energy and RKS XC matrix at sampled k-points.
    multigrid version of function pbc.dft.numint.nr_rks.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        exc : XC energy
        nelec : number of electrons obtained from the numerical integration
        veff : (nkpts, nao, nao) ndarray
            or list of veff if the input dm_kpts is a list of DMs
        vj : (nkpts, nao, nao) ndarray
            or list of vj if the input dm_kpts is a list of DMs
    '''
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

    cpu_df = multigrid.MultiGridFFTDF(cell)

    # density_on_G_mesh = new_evaluate_density_on_g_mesh(mydf, dm_kpts, hermi, kpts, derivative_order)
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
        fft.FFTDF.__init__(self, cell, kpts)
        sort_gaussian_pairs(self)


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
