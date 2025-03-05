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


def generate_ao_values(mydf, kpts):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    numerical_integrator = mydf._numint

    tasks = getattr(mydf, 'tasks', None)
    if tasks is None:
        tasks = multigrid.multi_grids_tasks(cell, mydf.mesh, log)
        mydf.tasks = tasks

    ao_values = []

    for grids_dense, grids_sparse in tasks:
        ao_values_for_this_task = {}
        ao_for_dense = numerical_integrator.eval_ao(grids_dense.cell, grids_dense.coords, kpts)
        ao_values_for_this_task["dense"] = ao_for_dense
        if grids_sparse is not None:
            ao_values_for_this_task["sparse"] = numerical_integrator.eval_ao(grids_sparse.cell,  grids_dense.coords, kpts)
        else:
            ao_values_for_this_task["sparse"] = None
        ao_values.append(ao_values_for_this_task)
        

    mydf.ao_values = ao_values


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
    for (grids_dense, grids_sparse), ao_values in zip(tasks, mydf.ao_values):

        mesh = grids_dense.mesh
        fft_grids = list(map(lambda mesh_points: np.fft.fftfreq(mesh_points, 1. / mesh_points).astype(np.int32), mesh))
        n_grid_points = np.prod(mesh)
        weight_per_grid_point = 1. / n_k_points * mydf.cell.vol / n_grid_points

        # The first pass handles all diffused functions using the regular
        # matrix multiplication code.
        density = cp.zeros((n_channels, density_slices, n_grid_points), dtype=cp.complex128)
        density_matrix_in_dense_region = dms[:, :, grids_dense.ao_idx[:, None], grids_dense.ao_idx]
        for k in range(n_k_points):
            for i in range(n_channels):
                if xc_type == 'LDA':
                    ao_dot_dm = cp.dot(ao_values["dense"][k], density_matrix_in_dense_region[i, k])
                    density_subblock = cp.einsum('xi,xi->x', ao_dot_dm, ao_values["dense"][k].conj())
                density[i, :] += density_subblock
            
        if grids_sparse is not None:
            density_matrix_slice = dms[:, :, grids_dense.ao_idx[:, None], grids_sparse.ao_idx]
            for k in range(n_k_points):
                for i in range(n_channels):
                    if xc_type == 'LDA':
                        ao_dot_dm = cp.dot(ao_values["dense"][k], density_matrix_slice[i, k])
                        density_subblock = cp.einsum('xi,xi->x', ao_dot_dm, ao_values["sparse"][k].conj())
                    density[i, :] += 2 * density_subblock
                    
        if hermi:
            density = density.real

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
    
    cpu_df = multigrid.FFTDF(cell, kpts)


    density_on_G_mesh = evaluate_density_on_g_mesh(mydf, dm_kpts, hermi, kpts, derivative_order)
    cpu_density_on_g_mesh = multigrid._eval_rhoG(cpu_df, dm_kpts, hermi, kpts, derivative_order)
    print(cp.allclose(density_on_G_mesh, cpu_density_on_g_mesh))
    
    assert 0
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
        generate_ao_values(self, kpts)


def fftdf(mf):
    mf.with_df, old_df = FFTDF(mf.cell), mf.with_df
    mf.with_df.__dict__.update(old_df.__dict__)
    return mf
