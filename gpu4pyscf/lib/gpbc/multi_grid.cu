#include <cstdio>
#include <cuda_runtime.h>
#include <gint/config.h>
#include <gint/gint.h>
#include <cassert>
#include <ios>
#include <gint/cuda_alloc.cuh>

#define atm(SLOT, I)     atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I)     bas[BAS_SLOTS * (I) + (SLOT)]

#define EIJ_CUTOFF       60

__host__ __device__ double distance_squared(const double x, const double y, const double z)
{
    return x * x + y * y + z * z;
}

__global__ void evaluate_xc_kernel_ss_orthogonal(
    double* fock, const double* xc_weights, const int i_shell_begin, const int i_shell_end,
    const int j_shell_begin, const int j_shell_end,
    const int* shell_to_function,
    const int n_images, const double* vectors_to_neighboring_images,
    const double a, const double b, const double c,
    const int offset_a, const int offset_b, const int offset_c,
    const int local_mesh_a, const int local_mesh_b, const int local_mesh_c,
    const int global_mesh_a, const int global_mesh_b, const int global_mesh_c,
    const int* atm, const int* bas, const double* env)
{
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
    const uint flattened_index = a_index * local_mesh_b * local_mesh_c + b_index * local_mesh_c + c_index;

    if (a_index >= local_mesh_a || b_index >= local_mesh_b || c_index >= local_mesh_c)
    {
        return;
    }

    const double position_x = a * (a_index + offset_a) / global_mesh_a;
    const double position_y = b * (b_index + offset_b) / global_mesh_b;
    const double position_z = c * (c_index + offset_c) / global_mesh_c;
    const int i_function_begin = shell_to_function[i_shell_begin];
    const int j_function_begin = shell_to_function[j_shell_begin];
    const int n_ao_i = shell_to_function[i_shell_end] - i_function_begin;
    const int n_ao_j = shell_to_function[j_shell_end] - j_function_begin;
    const int fock_stride = n_ao_i * n_ao_j;

    const double xc_value = xc_weights[flattened_index];

    for (int i_shell = i_shell_begin; i_shell < i_shell_end; i_shell++)
    {
        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset];
        const double i_y = env[i_coord_offset + 1];
        const double i_z = env[i_coord_offset + 2];
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];
        const int i_function_index = shell_to_function[i_shell] - i_function_begin;

        for (int j_shell = j_shell_begin; j_shell < j_shell_end; j_shell++)
        {
            const double j_exponent = env[bas(PTR_EXP, j_shell)];
            const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
            const double j_x = env[j_coord_offset];
            const double j_y = env[j_coord_offset + 1];
            const double j_z = env[j_coord_offset + 2];
            const double j_coeff = env[bas(PTR_COEFF, j_shell)];
            const int j_function_index = shell_to_function[j_shell] - j_function_begin;

            const double ij_exponent = i_exponent + j_exponent;
            for (int i_image = 0; i_image < n_images; i_image++)
            {
                const double i_image_x = vectors_to_neighboring_images[i_image * 3];
                const double i_image_y = vectors_to_neighboring_images[i_image * 3 + 1];
                const double i_image_z = vectors_to_neighboring_images[i_image * 3 + 2];

                const double shifted_i_x = i_x - i_image_x;
                const double shifted_i_y = i_y - i_image_y;
                const double shifted_i_z = i_z - i_image_z;

                const double ij_norm_squared = distance_squared(shifted_i_x - j_x,
                                                                shifted_i_y - j_y,
                                                                shifted_i_z - j_z);

                const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * ij_norm_squared;

                if (ij_exponent_in_prefactor > EIJ_CUTOFF)
                {
                    continue;
                }
                const double prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff * 0.282094791773878143 *
                    0.282094791773878143 * xc_value;

                const double x = position_x - (i_exponent * shifted_i_x + j_exponent * j_x) / ij_exponent;
                const double y = position_y - (i_exponent * shifted_i_y + j_exponent * j_y) / ij_exponent;
                const double z = position_z - (i_exponent * shifted_i_z + j_exponent * j_z) / ij_exponent;

                const double real_space_cutoff = sqrt(EIJ_CUTOFF / ij_exponent);

                const int lower_a_index = ceil((x - real_space_cutoff) / a);
                const int upper_a_index = floor((x + real_space_cutoff) / a);
                const int lower_b_index = ceil((y - real_space_cutoff) / b);
                const int upper_b_index = floor((y + real_space_cutoff) / b);
                const int lower_c_index = ceil((z - real_space_cutoff) / c);
                const int upper_c_index = floor((z + real_space_cutoff) / c);

                if (upper_a_index >= lower_a_index && upper_b_index >= lower_b_index &&
                    upper_c_index >= lower_c_index)
                {
                    double neighboring_gaussian_sum = 0;

                    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++)
                    {
                        for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++)
                        {
                            for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++)
                            {
                                const double r_squared = distance_squared(
                                    x - a_cell * a, y - b_cell * b, z - c_cell * c);

                                neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                            }
                        }
                    }

                    atomicAdd(fock + fock_stride * i_image + n_ao_i * j_function_index + i_function_index,
                              prefactor * neighboring_gaussian_sum);
                }
            }

            // TODO: def possible to reduce within a block first
        }
    }
}

__global__ void evaluate_density_new_kernel_ss(
    double* density, const double* density_matrices,
    const int* non_trivial_pairs, const int n_pairs,
    const int* i_shells, const int n_i_shells, const int* j_shells, const int n_j_shells,
    const int* shell_to_function, const int n_functions, const int* image_indices,
    const double* vectors_to_neighboring_images,
    const double* lattice_vectors, const double* reciprocal_lattice_vectors,
    const int mesh_a, const int mesh_b, const int mesh_c,
    const int* atm, const int* bas, const double* env)
{
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
    const int density_matrix_stride = n_i_shells * n_j_shells; // needs to fix to n_i_functions * n_j_functions

    if (a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c)
    {
        return;
    }

    const double position_x = lattice_vectors[0] * a_index / mesh_a + lattice_vectors[3] * b_index / mesh_b +
        lattice_vectors[6] * c_index / mesh_c;
    const double position_y = lattice_vectors[1] * a_index / mesh_a + lattice_vectors[4] * b_index / mesh_b +
        lattice_vectors[7] * c_index / mesh_c;
    const double position_z = lattice_vectors[2] * a_index / mesh_a + lattice_vectors[5] * b_index / mesh_b +
        lattice_vectors[8] * c_index / mesh_c;

    double density_value = 0;
    for (int i_pair = 0; i_pair < n_pairs; i_pair++)
    {
        const int image_index = image_indices[i_pair];

        const double image_x = vectors_to_neighboring_images[image_index * 3];
        const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
        const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

        const int pair_index = non_trivial_pairs[i_pair];
        const int j_shell_index = pair_index / n_i_shells;
        const int i_shell_index = pair_index % n_i_shells;
        const int i_shell = i_shells[i_shell_index];
        const int i_function = shell_to_function[i_shell] - shell_to_function[i_shells[0]];
        const int j_shell = j_shells[j_shell_index];
        const int j_function = shell_to_function[j_shell] - shell_to_function[j_shells[0]];

        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset] - image_x;
        const double i_y = env[i_coord_offset + 1] - image_y;
        const double i_z = env[i_coord_offset + 2] - image_z;
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];

        const double j_exponent = env[bas(PTR_EXP, j_shell)];
        const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
        const double j_x = env[j_coord_offset];
        const double j_y = env[j_coord_offset + 1];
        const double j_z = env[j_coord_offset + 2];
        const double j_coeff = env[bas(PTR_COEFF, j_shell)];

        const double ij_exponent = i_exponent + j_exponent;
        const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * distance_squared(
            i_x - j_x, i_y - j_y, i_z - j_z);
        const double density_matrix_value = density_matrices[image_index * density_matrix_stride +
            i_function * n_functions + j_function];
        const double prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff * 0.282094791773878143 *
            0.282094791773878143 * density_matrix_value;

        const double real_space_cutoff = sqrt(EIJ_CUTOFF / ij_exponent);

        const double x = position_x - (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
        const double y = position_y - (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
        const double z = position_z - (i_exponent * i_z + j_exponent * j_z) / ij_exponent;


        const int lower_a_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[0] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[3] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[6]);
        const int upper_a_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[0] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[3] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[6]);

        const int lower_b_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[1] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[4] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[7]);
        const int upper_b_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[1] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[4] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[7]);

        const int lower_c_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[2] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[5] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[8]);
        const int upper_c_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[2] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[5] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[8]);

        if (upper_a_index >= lower_a_index && upper_b_index >= lower_b_index &&
            upper_c_index >= lower_c_index)
        {
            double neighboring_gaussian_sum = 0;

            for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++)
            {
                for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++)
                {
                    for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++)
                    {
                        const double r_squared = distance_squared(
                            x - a_cell * lattice_vectors[0] - b_cell * lattice_vectors[3] -
                            c_cell * lattice_vectors[6],
                            y - a_cell * lattice_vectors[1] - b_cell * lattice_vectors[4] -
                            c_cell * lattice_vectors[7],
                            z - a_cell * lattice_vectors[2] - b_cell * lattice_vectors[5] -
                            c_cell * lattice_vectors[
                                8]);

                        neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                    }
                }
            }

            density_value += prefactor * neighboring_gaussian_sum;
        }
    }

    density[a_index * mesh_b * mesh_c + b_index * mesh_c + c_index] += density_value;
}

template <int n_channels>
__global__ void evaluate_density_new_kernel_ss(
    double* density, const double* density_matrices, const int* non_trivial_pairs,
    const int* i_shells, const int n_i_shells, const int* j_shells,
    const int* shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
    const int* sorted_pairs_per_local_grid, const int* accumulated_n_pairs_per_local_grid,
    const int* image_indices, const double* vectors_to_neighboring_images, const int n_images,
    const double* lattice_vectors, const double* reciprocal_lattice_vectors,
    const int mesh_a, const int mesh_b, const int mesh_c,
    const int* atm, const int* bas, const double* env)
{
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;
    const int density_matrix_stride = n_i_functions * n_j_functions; // needs to fix to n_i_functions * n_j_functions
    const int density_matrix_channel_stride = density_matrix_stride * n_images;

    if (a_index >= mesh_a || b_index >= mesh_b || c_index >= mesh_c)
    {
        return;
    }

    const double position_x = lattice_vectors[0] * a_index / mesh_a + lattice_vectors[3] * b_index / mesh_b +
        lattice_vectors[6] * c_index / mesh_c;
    const double position_y = lattice_vectors[1] * a_index / mesh_a + lattice_vectors[4] * b_index / mesh_b +
        lattice_vectors[7] * c_index / mesh_c;
    const double position_z = lattice_vectors[2] * a_index / mesh_a + lattice_vectors[5] * b_index / mesh_b +
        lattice_vectors[8] * c_index / mesh_c;

    double density_value[n_channels];
    double prefactor[n_channels];

#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++)
    {
        density_value[i_channel] = 0;
    }


    const uint local_grid_index = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    const int pairs_offset = accumulated_n_pairs_per_local_grid[local_grid_index];
    const int n_pairs_in_local_grid = accumulated_n_pairs_per_local_grid[local_grid_index + 1] - pairs_offset;

    for (int i_pair_index = 0; i_pair_index < n_pairs_in_local_grid; i_pair_index++)
    {
        const int i_pair = sorted_pairs_per_local_grid[pairs_offset + i_pair_index];
        const int image_index = image_indices[i_pair];

        const double image_x = vectors_to_neighboring_images[image_index * 3];
        const double image_y = vectors_to_neighboring_images[image_index * 3 + 1];
        const double image_z = vectors_to_neighboring_images[image_index * 3 + 2];

        const int pair_index = non_trivial_pairs[i_pair];
        const int j_shell_index = pair_index / n_i_shells;
        const int i_shell_index = pair_index % n_i_shells;
        const int i_shell = i_shells[i_shell_index];
        const int i_function = shell_to_ao_indices[i_shell];
        const int j_shell = j_shells[j_shell_index];
        const int j_function = shell_to_ao_indices[j_shell];

        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset] - image_x;
        const double i_y = env[i_coord_offset + 1] - image_y;
        const double i_z = env[i_coord_offset + 2] - image_z;
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];

        const double j_exponent = env[bas(PTR_EXP, j_shell)];
        const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
        const double j_x = env[j_coord_offset];
        const double j_y = env[j_coord_offset + 1];
        const double j_z = env[j_coord_offset + 2];
        const double j_coeff = env[bas(PTR_COEFF, j_shell)];

        const double ij_exponent = i_exponent + j_exponent;
        const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * distance_squared(
            i_x - j_x, i_y - j_y, i_z - j_z);

        const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff
            * 0.282094791773878143 * 0.282094791773878143;
#pragma unroll
        for (int i_channel = 0; i_channel < n_channels; i_channel++)
        {
            const double density_matrix_value = density_matrices[density_matrix_channel_stride * i_channel +
                image_index * density_matrix_stride + i_function * n_j_functions + j_function];
            prefactor[i_channel] = pair_prefactor * density_matrix_value;
        }
        const double real_space_cutoff = sqrt(EIJ_CUTOFF / ij_exponent);

        const double x = position_x - (i_exponent * i_x + j_exponent * j_x) / ij_exponent;
        const double y = position_y - (i_exponent * i_y + j_exponent * j_y) / ij_exponent;
        const double z = position_z - (i_exponent * i_z + j_exponent * j_z) / ij_exponent;


        const int lower_a_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[0] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[3] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[6]);
        const int upper_a_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[0] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[3] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[6]);

        const int lower_b_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[1] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[4] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[7]);
        const int upper_b_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[1] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[4] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[7]);

        const int lower_c_index = ceil((x - real_space_cutoff) * reciprocal_lattice_vectors[2] +
            (y - real_space_cutoff) * reciprocal_lattice_vectors[5] +
            (z - real_space_cutoff) *
            reciprocal_lattice_vectors[8]);
        const int upper_c_index = floor((x + real_space_cutoff) * reciprocal_lattice_vectors[2] +
            (y + real_space_cutoff) * reciprocal_lattice_vectors[5] +
            (z + real_space_cutoff) *
            reciprocal_lattice_vectors[8]);

        double neighboring_gaussian_sum = 0;

        for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++)
        {
            for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++)
            {
                for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++)
                {
                    const double r_squared = distance_squared(
                        x - a_cell * lattice_vectors[0] - b_cell * lattice_vectors[3] - c_cell * lattice_vectors[6],
                        y - a_cell * lattice_vectors[1] - b_cell * lattice_vectors[4] - c_cell * lattice_vectors[7],
                        z - a_cell * lattice_vectors[2] - b_cell * lattice_vectors[5] -
                        c_cell * lattice_vectors[8]);

                    neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                }
            }
        }

#pragma unroll
        for (int i_channel = 0; i_channel < n_channels; i_channel++)
        {
            density_value[i_channel] += prefactor[i_channel] * neighboring_gaussian_sum;
        }
    }
#pragma unroll
    for (int i_channel = 0; i_channel < n_channels; i_channel++)
    {
        density[i_channel * mesh_a * mesh_b * mesh_c
            + a_index * mesh_b * mesh_c + b_index * mesh_c + c_index] += density_value[i_channel];
    }
}


__global__ void evaluate_density_kernel_ss_orthogonal(
    double* density, const double* density_matrices,
    const int i_shell_begin, const int i_shell_end,
    const int j_shell_begin, const int j_shell_end,
    const int* shell_to_function,
    const int n_images, const double* vectors_to_neighboring_images,
    const double a, const double b, const double c,
    const int offset_a, const int offset_b, const int offset_c,
    const int local_mesh_a, const int local_mesh_b, const int local_mesh_c,
    const int global_mesh_a, const int global_mesh_b, const int global_mesh_c,
    const int* atm, const int* bas, const double* env)
{
    const uint a_index = threadIdx.z + blockDim.z * blockIdx.z;
    const uint b_index = threadIdx.y + blockDim.y * blockIdx.y;
    const uint c_index = threadIdx.x + blockDim.x * blockIdx.x;

    if (a_index >= local_mesh_a || b_index >= local_mesh_b || c_index >= local_mesh_c)
    {
        return;
    }

    const double position_x = a * (a_index + offset_a) / global_mesh_a;
    const double position_y = b * (b_index + offset_b) / global_mesh_b;
    const double position_z = c * (c_index + offset_c) / global_mesh_c;

    const int i_function_begin = shell_to_function[i_shell_begin];
    const int j_function_begin = shell_to_function[j_shell_begin];
    const int n_ao_i = shell_to_function[i_shell_end] - i_function_begin;
    const int n_ao_j = shell_to_function[j_shell_end] - j_function_begin;
    const int density_matrix_stride = n_ao_i * n_ao_j;
    double density_value = 0;

    for (int i_shell = i_shell_begin; i_shell < i_shell_end; i_shell++)
    {
        const double i_exponent = env[bas(PTR_EXP, i_shell)];
        const int i_coord_offset = atm(PTR_COORD, bas(ATOM_OF, i_shell));
        const double i_x = env[i_coord_offset];
        const double i_y = env[i_coord_offset + 1];
        const double i_z = env[i_coord_offset + 2];
        const double i_coeff = env[bas(PTR_COEFF, i_shell)];
        const int i_function_index = shell_to_function[i_shell] - i_function_begin;
        for (int j_shell = j_shell_begin; j_shell < j_shell_end; j_shell++)
        {
            const double j_exponent = env[bas(PTR_EXP, j_shell)];
            const int j_coord_offset = atm(PTR_COORD, bas(ATOM_OF, j_shell));
            const double j_x = env[j_coord_offset];
            const double j_y = env[j_coord_offset + 1];
            const double j_z = env[j_coord_offset + 2];
            const double j_coeff = env[bas(PTR_COEFF, j_shell)];
            const int j_function_index = shell_to_function[j_shell] - j_function_begin;

            const double ij_exponent = i_exponent + j_exponent;

            for (int i_image = 0; i_image < n_images; i_image++)
            {
                const double i_image_x = vectors_to_neighboring_images[i_image * 3];
                const double i_image_y = vectors_to_neighboring_images[i_image * 3 + 1];
                const double i_image_z = vectors_to_neighboring_images[i_image * 3 + 2];

                const double shifted_i_x = i_x - i_image_x;
                const double shifted_i_y = i_y - i_image_y;
                const double shifted_i_z = i_z - i_image_z;

                const double ij_norm_squared = distance_squared(shifted_i_x - j_x,
                                                                shifted_i_y - j_y,
                                                                shifted_i_z - j_z);

                const double ij_exponent_in_prefactor = i_exponent * j_exponent / ij_exponent * ij_norm_squared;

                if (ij_exponent_in_prefactor > EIJ_CUTOFF)
                {
                    continue;
                }

                const double density_matrix_value = density_matrices[i_image * density_matrix_stride +
                    j_function_index * n_ao_i + i_function_index];

                const double pair_prefactor = exp(-ij_exponent_in_prefactor) * i_coeff * j_coeff
                    * 0.282094791773878143 * 0.282094791773878143;
                const double prefactor = pair_prefactor * density_matrix_value;

                const double x = position_x - (i_exponent * shifted_i_x + j_exponent * j_x) / ij_exponent;
                const double y = position_y - (i_exponent * shifted_i_y + j_exponent * j_y) / ij_exponent;
                const double z = position_z - (i_exponent * shifted_i_z + j_exponent * j_z) / ij_exponent;

                const double real_space_cutoff = sqrt(EIJ_CUTOFF / ij_exponent);

                const int lower_a_index = ceil((x - real_space_cutoff) / a);
                const int upper_a_index = floor((x + real_space_cutoff) / a);
                const int lower_b_index = ceil((y - real_space_cutoff) / b);
                const int upper_b_index = floor((y + real_space_cutoff) / b);
                const int lower_c_index = ceil((z - real_space_cutoff) / c);
                const int upper_c_index = floor((z + real_space_cutoff) / c);


                if (upper_a_index >= lower_a_index && upper_b_index >= lower_b_index &&
                    upper_c_index >= lower_c_index)
                {
                    double neighboring_gaussian_sum = 0;

                    for (int a_cell = lower_a_index; a_cell <= upper_a_index; a_cell++)
                    {
                        for (int b_cell = lower_b_index; b_cell <= upper_b_index; b_cell++)
                        {
                            for (int c_cell = lower_c_index; c_cell <= upper_c_index; c_cell++)
                            {
                                const double r_squared = distance_squared(
                                    x - a_cell * a, y - b_cell * b, z - c_cell * c);

                                neighboring_gaussian_sum += exp(-ij_exponent * r_squared);
                            }
                        }
                    }

                    density_value += prefactor * neighboring_gaussian_sum;
                }
            }
        }
    }

    density[a_index * local_mesh_b * local_mesh_c + b_index * local_mesh_c + c_index] += density_value;
}

template <int n_channels>
void new_evaluate_density_driver(
    double* density, const double* density_matrices,
    const int left_angular, const int right_angular, const int* non_trivial_pairs,
    const int* i_shells, const int n_i_shells, const int* j_shells,
    const int* shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
    const int* sorted_pairs_per_local_grid, const int* accumulated_n_pairs_per_local_grid,
    const int* image_indices, const double* vectors_to_neighboring_images, const int n_images,
    const double* lattice_vectors, const double* reciprocal_lattice_vectors,
    const int* mesh, const int* atm, const int* bas, const double* env, const int blocking_sizes[3])
{
    dim3 block_size(blocking_sizes[2], blocking_sizes[1], blocking_sizes[0]);
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    dim3 block_grid((mesh_c + blocking_sizes[2] - 1) / blocking_sizes[2],
                    (mesh_b + blocking_sizes[1] - 1) / blocking_sizes[1],
                    (mesh_a + blocking_sizes[0] - 1) / blocking_sizes[0]);
    if (left_angular == 0 && right_angular == 0)
    {
        evaluate_density_new_kernel_ss<n_channels><<<block_grid, block_size>>>(
            density, density_matrices, non_trivial_pairs, i_shells, n_i_shells, j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
            lattice_vectors, reciprocal_lattice_vectors, mesh_a, mesh_b, mesh_c, atm, bas, env);
    }
    else
    {
        fprintf(stderr,
                "angular momentum pair %d, %d is not supported in new_evaluate_density_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

extern "C" {
void evaluate_xc_driver(
    double* fock, const double* xc_weights, const int* shells_slice, const int* shell_to_function,
    const int left_angular, const int right_angular,
    const int n_images, const double* vectors_to_neighboring_images,
    const double* lattice_vector, const int* offset, const int* local_mesh, const int* global_mesh,
    const int* atm, const int* bas, const double* env)
{
    dim3 block_size(4, 4, 4);
    int local_mesh_a = local_mesh[0];
    int local_mesh_b = local_mesh[1];
    int local_mesh_c = local_mesh[2];
    dim3 block_grid((local_mesh_c + 3) / 4, (local_mesh_b + 3) / 4, (local_mesh_a + 3) / 4);

    const double a = lattice_vector[0];
    const double b = lattice_vector[4];
    const double c = lattice_vector[8];

    if (left_angular == 0 && right_angular == 0)
    {
        evaluate_xc_kernel_ss_orthogonal<<<block_grid, block_size>>>(fock, xc_weights, shells_slice[0],
                                                                     shells_slice[1], shells_slice[2],
                                                                     shells_slice[3],
                                                                     shell_to_function,
                                                                     n_images, vectors_to_neighboring_images,
                                                                     a, b, c,
                                                                     offset[0], offset[1], offset[2],
                                                                     local_mesh_a, local_mesh_b, local_mesh_c,
                                                                     global_mesh[0], global_mesh[1], global_mesh[2],
                                                                     atm, bas, env);
    }
    else
    {
        fprintf(stderr, "angular momentum pair %d, %d is not supported in evaluate_xc_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

void evaluate_density_driver(
    double* density, const double* density_matrices, const int* shells_slice, const int* shell_to_function,
    const int left_angular, const int right_angular,
    const int n_images, const double* vectors_to_neighboring_images,
    const double* lattice_vector, const int* offset, const int* local_mesh, const int* global_mesh,
    const int* atm, const int* bas, const double* env)
{
    dim3 block_size(4, 4, 4);
    int local_mesh_a = local_mesh[0];
    int local_mesh_b = local_mesh[1];
    int local_mesh_c = local_mesh[2];
    dim3 block_grid((local_mesh_c + 3) / 4, (local_mesh_b + 3) / 4, (local_mesh_a + 3) / 4);

    const double a = lattice_vector[0];
    const double b = lattice_vector[4];
    const double c = lattice_vector[8];

    if (left_angular == 0 && right_angular == 0)
    {
        evaluate_density_kernel_ss_orthogonal<<<block_grid, block_size>>>(density, density_matrices, shells_slice[0],
                                                                          shells_slice[1], shells_slice[2],
                                                                          shells_slice[3],
                                                                          shell_to_function,
                                                                          n_images, vectors_to_neighboring_images,
                                                                          a, b, c,
                                                                          offset[0], offset[1], offset[2],
                                                                          local_mesh_a, local_mesh_b, local_mesh_c,
                                                                          global_mesh[0], global_mesh[1],
                                                                          global_mesh[2],
                                                                          atm, bas, env);
    }
    else
    {
        fprintf(stderr, "angular momentum pair %d, %d is not supported in evaluate_density_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

void new_evaluate_density_driver(
    double* density, const double* density_matrices, const int left_angular, const int right_angular,
    const int* non_trivial_pairs, const int n_pairs,
    const int* i_shells, const int n_i_shells, const int* j_shells, const int n_j_shells,
    const int* shell_to_function, const int n_functions,
    const int* image_indices, const double* vectors_to_neighboring_images,
    const double* lattice_vectors, const double* reciprocal_lattice_vectors,
    const int* mesh, const int* atm, const int* bas, const double* env)
{
    dim3 block_size(4, 4, 4);
    int mesh_a = mesh[0];
    int mesh_b = mesh[1];
    int mesh_c = mesh[2];
    dim3 block_grid((mesh_c + 3) / 4, (mesh_b + 3) / 4, (mesh_a + 3) / 4);


    if (left_angular == 0 && right_angular == 0)
    {
        evaluate_density_new_kernel_ss<<<block_grid, block_size>>>(density, density_matrices, non_trivial_pairs,
                                                                   n_pairs,
                                                                   i_shells, n_i_shells, j_shells, n_j_shells,
                                                                   shell_to_function, n_functions,
                                                                   image_indices, vectors_to_neighboring_images,
                                                                   lattice_vectors, reciprocal_lattice_vectors, mesh_a,
                                                                   mesh_b, mesh_c, atm, bas, env);
    }
    else
    {
        fprintf(stderr, "angular momentum pair %d, %d is not supported in new_evaluate_density_driver\n", left_angular,
                right_angular);
    }

    checkCudaErrors(cudaPeekAtLastError());
}

void new_evaluate_density_driver_with_local_sort(
    double* density, const double* density_matrices,
    const int left_angular, const int right_angular, const int* non_trivial_pairs,
    const int* i_shells, const int n_i_shells, const int* j_shells,
    const int* shell_to_ao_indices, const int n_i_functions, const int n_j_functions,
    const int* sorted_pairs_per_local_grid, const int* accumulated_n_pairs_per_local_grid,
    const int* image_indices, const double* vectors_to_neighboring_images, const int n_images,
    const double* lattice_vectors, const double* reciprocal_lattice_vectors,
    const int* mesh, const int* atm, const int* bas, const double* env, const int blocking_sizes[3],
    const int n_channels)
{
    if (n_channels == 1)
    {
        new_evaluate_density_driver<1>(
            density, density_matrices, left_angular, right_angular, non_trivial_pairs, i_shells, n_i_shells, j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
            lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    }
    else if (n_channels == 2)
    {
        new_evaluate_density_driver<2>(
            density, density_matrices, left_angular, right_angular, non_trivial_pairs, i_shells, n_i_shells, j_shells,
            shell_to_ao_indices, n_i_functions, n_j_functions, sorted_pairs_per_local_grid,
            accumulated_n_pairs_per_local_grid, image_indices, vectors_to_neighboring_images, n_images,
            lattice_vectors, reciprocal_lattice_vectors, mesh, atm, bas, env, blocking_sizes);
    }
    else
    {
        fprintf(stderr, "n_channels more than 2 is not supported.\n");
    }
}
}
