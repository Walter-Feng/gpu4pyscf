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
import cupy as cp
import scipy

import ctypes

libgdft = cupy_helper.load_library('libgdft')
libgpbc = cupy_helper.load_library('libgpbc')


def eval_mat(cell, weights, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, mesh=None, offset=None, lattice_sum_mesh=None):
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(mesh) / vol
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
            weights = weights.reshape(-1, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    lattice_vector = cell.lattice_vectors()
    b = np.linalg.inv(lattice_vector.T)
    if offset is None:
        offset = (0, 0, 0)
    if lattice_sum_mesh is None:
        lattice_sum_mesh = mesh
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
            ctypes.c_int(comp), ctypes.c_int(hermi),
            (ctypes.c_int * 4)(i0, i1, j0, j1),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(log_prec),
            ctypes.c_int(cell.dimension),
            ctypes.c_int(nimgs),
            vectors_to_neighboring_images.ctypes.data_as(ctypes.c_void_p),
            lattice_vector.ctypes.data_as(ctypes.c_void_p),
            b.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*lattice_sum_mesh),
            (ctypes.c_int * 3)(*mesh),
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
            bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(fock)

    def new_driver_wrapper(xc_weights):
        fock = cp.zeros((nimgs, comp, naoj, naoi))
        assert isinstance(xc_weights, cp.ndarray)
        new_driver(ctypes.cast(mat.data.ptr, ctypes.c_void_p),
                   ctypes.cast(xc_weights.data.ptr, ctypes.c_void_p),
                   (ctypes.c_int * 4)(i0, i1, j0, j1),
                   ctypes.cast(ao_loc_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.c_int(0), ctypes.c_int(0),
                   ctypes.c_int(nimgs),
                   ctypes.cast(vectors_to_neighboring_images_on_gpu.data.ptr, ctypes.c_void_p),
                   lattice_vector.ctypes.data_as(ctypes.c_void_p),
                   (ctypes.c_int * 3)(*offset),
                   (ctypes.c_int * 3)(*lattice_sum_mesh),
                   ctypes.cast(atm_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(bas_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(env_on_gpu.data.ptr, ctypes.c_void_p))

        return fock

    out = []

    cpu = make_mat(weights)
    gpu = new_driver_wrapper(weights)
    print("cpu max: ", np.amax(cpu))
    print("gpu max: ", cp.amax(gpu))
    print("diff: ", cp.amax((cp.asarray(cpu) - gpu)))

    assert 0
    for wv in weights:
        if cell.dimension == 0:
            mat = make_mat(wv)[0].transpose(0, 2, 1)
            if hermi == 1:
                for i in range(comp):
                    cupy_helper.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
        elif kpts is None or multigrid.gamma_point(kpts):
            mat = make_mat(wv).sum(axis=0).transpose(0, 2, 1)
            if hermi == 1:
                for i in range(comp):
                    cupy_helper.hermi_triu(mat[i], inplace=True)
            if comp == 1:
                mat = mat[0]
            if getattr(kpts, 'ndim', None) == 2:
                mat = mat[None, :]
        else:
            mat = make_mat(wv)
            expkL = cp.exp(1j * kpts.reshape(-1, 3).dot(vectors_to_neighboring_images.T))
            mat = cp.einsum('kr,rcij->kcij', expkL, mat)
            if hermi == 1:
                for i in range(comp):
                    for k in range(len(kpts)):
                        cp.hermi_triu(mat[k, i], inplace=True)
            mat = mat.transpose(0, 1, 3, 2)
            if comp == 1:
                mat = mat[:, 0]
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out


def evaluate_density(cell, dm, shells_slice=None, hermi=0, xc_type='LDA', kpts=None,
                     mesh=None, offset=None, lattice_sum_mesh=None, ignore_imag=False,
                     out=None):
    '''Collocate the *real* density (opt. gradients) on the real-space grid.

    Kwargs:
        ignore_image :
            The output density is assumed to be real if ignore_imag=True.
    '''
    assert (all(cell._bas[:, multigrid.NPRIM_OF] == 1))
    if mesh is None:
        mesh = cell.mesh
    vol = cell.vol
    weight_penalty = np.prod(mesh) / vol
    minimum_exponent = np.hstack(cell.bas_exps()).min()
    theta_ij = minimum_exponent / 2
    lattice_summation_factor = max(2 * np.pi * cell.rcut / (vol * theta_ij), 1)
    precision = cell.precision / weight_penalty / lattice_summation_factor
    if xc_type != 'LDA':
        precision *= .1
    # concatenate two molecules
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    env[multigrid.PTR_EXPDROP] = min(precision * multigrid.EXTRA_PREC, multigrid.EXPDROP)
    ao_loc = gto.moleintor.make_loc(bas, 'cart')
    if shells_slice is None:
        shells_slice = (0, cell.nbas, 0, cell.nbas)
    i0, i1, j0, j1 = shells_slice
    if hermi == 1:
        assert (i0 == j0 and i1 == j1)
    j0 += cell.nbas
    j1 += cell.nbas
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    dm = cp.asarray(dm, order='C')
    assert (dm.shape[-2:] == (naoi, naoj))
    print("nao pair: ", (naoi, naoj))

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
    if lattice_sum_mesh is None:
        lattice_sum_mesh = mesh
    precision_in_log = np.log(precision * multigrid.EXTRA_PREC)

    if abs(lattice_vector - np.diag(lattice_vector.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    xc_type = xc_type.upper()
    if xc_type == 'LDA':
        n_components = 1
    elif xc_type == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        n_components = 4
    else:
        raise NotImplementedError('meta-GGA')
    if n_components == 1:
        shape = (np.prod(lattice_sum_mesh),)
    else:
        shape = (n_components, np.prod(lattice_sum_mesh))
    kernel_name = 'NUMINTrho_' + xc_type.lower() + lattice_type
    driver = libdft.NUMINT_rho_drv

    new_driver = libgpbc.evaluate_density_driver

    def driver_wrapper(density_shape, dm, hermi):
        density = np.zeros(density_shape, order='C')
        driver(getattr(libdft, kernel_name),
               density.ctypes.data_as(ctypes.c_void_p),
               dm.get().ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(n_components), ctypes.c_int(hermi),
               (ctypes.c_int * 4)(i0, i1, j0, j1),
               ao_loc.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_double(precision_in_log),
               ctypes.c_int(cell.dimension),
               ctypes.c_int(n_images),
               vectors_to_neighboring_images.ctypes.data_as(ctypes.c_void_p),
               lattice_vector.ctypes.data_as(ctypes.c_void_p),
               reciprocal_lattice_vector.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int * 3)(*offset), (ctypes.c_int * 3)(*lattice_sum_mesh),
               (ctypes.c_int * 3)(*mesh),
               atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
               bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
               env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(env)))
        return cp.asarray(density)

    atm_on_gpu = cp.asarray(atm)
    bas_on_gpu = cp.asarray(bas)
    env_on_gpu = cp.asarray(env)
    ao_loc_on_gpu = cp.asarray(ao_loc)
    vectors_to_neighboring_images_on_gpu = cp.asarray(vectors_to_neighboring_images)

    def new_driver_wrapper(density, dm):
        assert isinstance(density, cp.ndarray)
        new_driver(ctypes.cast(density.data.ptr, ctypes.c_void_p),
                   ctypes.cast(dm.data.ptr, ctypes.c_void_p),
                   (ctypes.c_int * 4)(i0, i1, j0, j1),
                   ctypes.cast(ao_loc_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.c_int(0), ctypes.c_int(0),
                   ctypes.c_int(n_images),
                   ctypes.cast(vectors_to_neighboring_images_on_gpu.data.ptr, ctypes.c_void_p),
                   lattice_vector.ctypes.data_as(ctypes.c_void_p),
                   (ctypes.c_int * 3)(*offset),
                   (ctypes.c_int * 3)(*lattice_sum_mesh),
                   ctypes.cast(atm_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(bas_on_gpu.data.ptr, ctypes.c_void_p),
                   ctypes.cast(env_on_gpu.data.ptr, ctypes.c_void_p))

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
                has_imag = (hermi == 0 and abs(dmI).max() > 1e-8)
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
                rho_i = cp.empty(shape, cp.complex128)
                rho_i.real = driver_wrapper(shape, dmR, 0)
                rho_i.imag = driver_wrapper(shape, dmI, 0)
            else:
                assert out[i].dtype == cp.complex128
                rho_i = out[i].reshape(shape)
                rho_i.real += driver_wrapper(shape, dmR, 0)
                rho_i.imag += driver_wrapper(shape, dmI, 0)
        else:
            if out is None:
                # rho_i needs to be initialized to 0 because rho_i is updated
                # inplace in function NUMINT_rho_drv
                # rho_i = driver_wrapper(shape, dmR, hermi)
                rho_i = driver_wrapper(shape, dmR, hermi)
            else:
                assert out[i].dtype == cp.double
                rho_i = out[i].reshape(shape)
                rho_i += driver_wrapper(shape, dmR, hermi)

        dmR = dmI = None
        rho.append(rho_i)

    if n_dm == 1:
        rho = rho[0]
    return cp.asarray(rho)


def _eval_rho_bra(cell, dms, shell_ranges, hermi, xc_type, kpts, grids, ignore_imag, log):
    lattice_vectors = np.asarray(cell.lattice_vectors())
    max_element = lattice_vectors.max()
    mesh = np.asarray(grids.mesh)
    real_space_cutoff = grids.cell.rcut
    nset = dms.shape[0]
    if xc_type == 'LDA':
        rho_slices = 1
    else:
        rho_slices = 4

    if real_space_cutoff > max_element * multigrid.R_RATIO_SUBLOOP:
        density = evaluate_density(cell, dms, shell_ranges, hermi, xc_type, kpts, mesh, ignore_imag=ignore_imag)
        return cp.reshape(density, (nset, rho_slices, np.prod(mesh)))

    if hermi == 1 or ignore_imag:
        density = cp.zeros((nset, rho_slices) + tuple(mesh))
    else:
        density = cp.zeros((nset, rho_slices) + tuple(mesh), dtype=cp.complex128)

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
        lattice_sum_begin = b.dot(atom_position - real_space_cutoff)
        lattice_sum_end = b.dot(atom_position + real_space_cutoff)

        if (np.all(0 < lattice_sum_begin) and np.all(lattice_sum_end < 1)):
            copied_cell._bas = np.vstack((row_basis, cell._bas[col_shell_begin:col_shell_end]))
            n_row_shells = len(row_shells)
            sub_slice = (0, n_row_shells, n_row_shells, n_row_shells + n_col_shells)

            head = (lattice_sum_begin * mesh).astype(int)
            tail = np.ceil(lattice_sum_end * mesh).astype(int)
            lattice_sum_mesh = tail - head
            log.debug1('atm %d  rcut %f  offset %s submesh %s',
                       atom, real_space_cutoff, head, lattice_sum_mesh)
            evaluated_density = evaluate_density(copied_cell, density_matrix_subblock, sub_slice, hermi, xc_type, kpts,
                                                 mesh, head, lattice_sum_mesh, ignore_imag=ignore_imag)
            #:rho[:,:,offset[0]:mesh1[0],offset[1]:mesh1[1],offset[2]:mesh1[2]] += \
            #:        numpy.reshape(rho1, (nset, rhodim) + tuple(submesh))
            gx = cp.arange(head[0], tail[0], dtype=cp.int32)
            gy = cp.arange(head[1], tail[1], dtype=cp.int32)
            gz = cp.arange(head[2], tail[2], dtype=cp.int32)
            print("mesh: ", mesh)
            print("lattice_sum_mesh: ", lattice_sum_mesh)

            density[cp.ix_(cp.arange(nset), cp.arange(rho_slices), gx, gy, gz)] += evaluated_density.reshape(
                (-1,) + lattice_sum_mesh)
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
        evaluate_density(copied_cell, density_matrix_subblock, sub_slice, hermi, xc_type, kpts,
                         mesh, ignore_imag=ignore_imag, out=density)
    return density.reshape((nset, rho_slices, np.prod(mesh)))


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
    skip_count = 0
    skip_threshold = 3
    # print(len(tasks))
    for grids_dense, grids_sparse in tasks:
        # if skip_count < skip_threshold:
        #     skip_count += 1
        #     continue
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
            equivalent_cell_in_dense, primitive_coeff_in_dense = subcell_in_dense_region.decontract_basis(to_cart=True,
                                                                                                          aggregate=True)
            equivalent_cell_in_sparse, primitive_coeff_in_sparse = subcell_in_sparse_region.decontract_basis(
                to_cart=True, aggregate=True)
            concatenated_cell = equivalent_cell_in_dense + equivalent_cell_in_sparse
            concatenated_coeff = scipy.linalg.block_diag(primitive_coeff_in_dense, primitive_coeff_in_sparse)

            n_primitive_gtos_in_dense = multigrid._pgto_shells(subcell_in_dense_region)
            n_primitive_gtos_in_two_regions = multigrid._pgto_shells(concatenated_cell)

            if deriv == 0:
                if hermi:
                    n_ao_in_sparse, n_ao_in_dense = density_matrix_block_from_sparse_rows.shape[2:]
                    density_matrix_block_from_dense_rows[:, :, :,
                    n_ao_in_dense:] += density_matrix_block_from_sparse_rows.transpose(0, 1, 3, 2)
                    coeff_sandwiched_density_matrix = cp.einsum('nkij,pi,qj->nkpq',
                                                                density_matrix_block_from_dense_rows,
                                                                primitive_coeff_in_dense, concatenated_coeff)
                    shells_slice = (0, n_primitive_gtos_in_dense, 0, n_primitive_gtos_in_two_regions)
                    #:rho = eval_rho(t_cell, pgto_dms, shls_slice, 0, 'LDA', kpts,
                    #:               offset=None, submesh=None, ignore_imag=True)
                    density = _eval_rho_bra(concatenated_cell, coeff_sandwiched_density_matrix, shells_slice, 0,
                                            'LDA', kpts, grids_dense, True, log)
                else:
                    raise NotImplementedError

            elif deriv == 1:
                raise NotImplementedError

        weight_per_grid_point = 1. / n_k_points * cell.vol / n_grid_points
        density_contribution_on_g_mesh = tools.fft(density.reshape(n_channels * density_slices, -1), mesh)
        density_contribution_on_g_mesh *= weight_per_grid_point
        fft_grids = map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh)
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
            veff_slice = eval_mat(concatenated_cell, veff_real_part, shells_indices, 1, 0, 'LDA', kpts)
            # Imaginary part may contribute
            if not ignore_vG_imag:
                veff_slice += eval_mat(concatenated_cell, veff_imag_part, shells_indices, 1, 0, 'LDA', kpts) * 1j

            veff_slice = cp.einsum('nkpq,pi,qj->nkij', cp.asarray(veff_slice),
                                   cp.asarray(decontracted_coeff_in_dense), cp.asarray(concatenated_coeff))

            vj_kpts[:, :, ao_index_in_dense[:, None], ao_index_in_dense] += veff_slice[:, :, :, :n_ao_in_sparse]
            vj_kpts[:, :, ao_index_in_dense[:, None], ao_index_in_sparse] += veff_slice[:, :, :, n_ao_in_sparse:]

            if hermi == 1:
                vj_kpts[:, :, ao_index_in_sparse[:, None], ao_index_in_dense] += \
                    veff_slice[:, :, :, n_ao_in_sparse:].transpose(0, 1, 3, 2).conj()
            else:
                shells_indices = (n_shells_in_dense, n_shells_in_total, 0, n_shells_in_dense)
                veff_slice = eval_mat(concatenated_cell, veff_real_part, shells_indices, 1, 0, 'LDA', kpts)
                # Imaginary part may contribute
                if not ignore_vG_imag:
                    veff_slice += eval_mat(concatenated_cell, veff_imag_part, shells_indices, 1, 0, 'LDA', kpts) * 1j
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

    density_on_G_mesh = evaluate_density_on_g_mesh(mydf, dm_kpts, hermi, kpts, derivative_order)

    mesh = mydf.mesh
    ngrids = np.prod(mesh)
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

    weighted_xc_for_fock_on_g_mesh = cp.ndarray((nset, *density_in_real_space.shape))
    xc_energy_sum = cp.zeros(nset)
    for i in range(nset):
        if xc_type == 'LDA':
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i, 0], deriv=1,
                                                                          xctype=xc_type)[:2]
        else:
            xc_for_energy, xc_for_fock = numerical_integrator.eval_xc_eff(xc_code, density_in_real_space[i], deriv=1,
                                                                          xctype=xc_type)[:2]

        xc_energy_sum[i] += (density_in_real_space[i, 0] * xc_for_energy.flatten()).sum() * weight
        weighted_xc_for_fock_on_g_mesh[i] = tools.fft(weight * xc_for_fock, mesh)
    density_in_real_space = density_on_G_mesh = None

    if nset == 1:
        coulomb_energy = coulomb_energy[0]
        n_electrons = n_electrons[0]
        xc_energy_sum = xc_energy_sum[0]
    log.debug('Multigrid exc %s  nelec %s', xc_energy_sum, n_electrons)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    if xc_type == 'LDA':
        if with_j:
            weighted_xc_for_fock_on_g_mesh[:, 0] += coulomb_on_g_mesh.reshape(nset, *mesh)
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


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
