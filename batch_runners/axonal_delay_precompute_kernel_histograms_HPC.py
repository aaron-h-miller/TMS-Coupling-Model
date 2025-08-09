import argparse
import os
import numpy as np
import h5py
import multiprocessing
import pathlib
from postprocessing_modules.axonal_delay_kernel import get_axonal_delay_kernel
from _functools import partial
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


def chunk_handler(indices, parameters, cell_type_labels):
    global Z_STEP
    global T_STEP
    global DENSITY
    global SMOOTH_T_STEP
    global SMOOTH_Z_STEP
    global chunk_dir
    chunk_path = chunk_dir.joinpath(f'axon_kernel_chunk_{indices[0]}_{indices[-1]}.hdf5')
    idx_start = indices[0]
    if chunk_path.is_file():
        with h5py.File(chunk_path, 'r') as f_chunk:
            last_index = int(f_chunk.attrs.__getitem__('last_index'))
        if last_index < indices[-1]:
            idx_start = last_index + 1
        else:
            idx_start = indices[-1] + 1

    for index in range(idx_start, indices[-1] + 1):
        if index % 100 == 0:
            print(f'{index} | Chunk {indices[0]} / {indices[-1]} {(index - indices[0]) / (indices[-1] - indices[0]):.0%}')
        theta, gradient, intensity = parameters[index, :]
        cell_type = cell_type_labels[index]

        kernel = get_axonal_delay_kernel(source_layer_name=cell_type, delay_z_folder=delays_dir,
                                         precomputed_kernels_folder=kernels_dir, theta=theta, gradient=gradient,
                                         intensity=intensity, z_step=Z_STEP, t_step=T_STEP, smooth_z_step=SMOOTH_Z_STEP,
                                         smooth_t_step=SMOOTH_T_STEP, density=DENSITY, compute_fresh=True)

        with h5py.File(chunk_path, 'a') as f_chunk:
            f_chunk.create_dataset(name=f'cell_histogram_{index}', data=kernel.cell_histogram, dtype='float32')
            f_chunk.create_dataset(name=f'layer_histogram_{index}', data=kernel.layer_histogram, dtype='float32')
            f_chunk.create_dataset(name=f'cell_smoothed_{index}', data=kernel.cell_kernel, dtype='float32')
            f_chunk.create_dataset(name=f'layer_smoothed_{index}', data=kernel.layer_kernel, dtype='float32')
            f_chunk.attrs.create(name='last_index', data=index)
            f_chunk.create_dataset(name=f't_bin_centers_{index}', data=kernel.t_bin_centers, dtype='float32')
            f_chunk.create_dataset(name=f't_bins_{index}', data=kernel.t_bins, dtype='float32')
            f_chunk.create_dataset(name=f't_kernel_{index}', data=kernel.t_kernel, dtype='float32')
            f_chunk.create_dataset(name=f'z_bins_{index}', data=kernel.z_bins, dtype='float32')
            f_chunk.create_dataset(name=f'z_bins_layer_{index}', data=kernel.z_bins_layer, dtype='float32')
            f_chunk.create_dataset(name=f'z_bin_centers_{index}', data=kernel.z_bin_centers, dtype='float32')
            f_chunk.create_dataset(name=f'z_bin_layer_centers_{index}', data=kernel.z_bin_layer_centers, dtype='float32')
            f_chunk.create_dataset(name=f'z_kernel_{index}', data=kernel.z_kernel, dtype='float32')
            f_chunk.create_dataset(name=f'z_layer_{index}', data=kernel.z_layer, dtype='float32')
            f_chunk.create_dataset(name=f'z_layer_edges_{index}', data=kernel.z_layer_edges, dtype='float32')
            f_chunk.create_dataset(name=f'density_{index}', data=kernel.density, dtype='bool')

    print(f'Chunk {indices[0]} / {indices[-1]} Complete')


def merge_precomputed_kernel_chunks(chunk_folder: pathlib.Path, all_params, cell_types):
    assert all_params.shape[0] == len(cell_types), 'Parameters and cell type labels should match length'
    all_chunk_paths = sorted(chunk_folder.iterdir(), key=lambda pth: int(pth.stem.split('_')[-2]))
    chunk_indices = np.vstack([[int(pth.stem.split('_')[-2]), int(pth.stem.split('_')[-1])] for pth in all_chunk_paths])

    for cell_type in list(set(cell_types)):
        print(f'Merging for Cell Type: {cell_type}')
        t_label = f'{T_STEP:.3}'.replace('.', 'p')
        if DENSITY:
            dens_label = 'density'
        else:
            dens_label = 'histogram'
        cell_type_file = kernels_dir.joinpath(f'axonal_delay_kernels_{cell_type}_zstep_{Z_STEP}_tstep_{t_label}_{dens_label}.hdf5')
        first_idx = np.where(np.array(cell_types) == cell_type)[0][0]
        last_idx = np.where(np.array(cell_types) == cell_type)[0][-1]

        first_chunk = np.where((first_idx >= chunk_indices[:, 0]) & (first_idx <= chunk_indices[:, 1]))[0][0]
        last_chunk = np.where((last_idx >= chunk_indices[:, 0]) & (last_idx <= chunk_indices[:, 1]))[0][0]

        # Retrieve constant params/data series for this cell type
        first_chunk_iter_start = max(chunk_indices[first_chunk, 0], first_idx)
        with h5py.File(all_chunk_paths[first_chunk], 'r') as f_cell_type:
            t_bin_centers = f_cell_type[f't_bin_centers_{first_chunk_iter_start}'][:]
            t_bins = f_cell_type[f't_bins_{first_chunk_iter_start}'][:]
            t_kernel = f_cell_type[f't_kernel_{first_chunk_iter_start}'][:]
            z_bins = f_cell_type[f'z_bins_{first_chunk_iter_start}'][:]
            z_bins_layer = f_cell_type[f'z_bins_layer_{first_chunk_iter_start}'][:]
            z_bin_centers = f_cell_type[f'z_bin_centers_{first_chunk_iter_start}'][:]
            z_bin_layer_centers = f_cell_type[f'z_bin_layer_centers_{first_chunk_iter_start}'][:]
            z_kernel = f_cell_type[f'z_kernel_{first_chunk_iter_start}'][:]
            z_layer = f_cell_type[f'z_layer_{first_chunk_iter_start}'][:]
            z_layer_edges = f_cell_type[f'z_layer_edges_{first_chunk_iter_start}'][:]
            density = f_cell_type[f'density_{first_chunk_iter_start}'][()]
        # Initialize cell type file
        with h5py.File(cell_type_file, 'w') as f_save:
            f_save.create_dataset(
                name='cell_histograms',
                shape=(last_idx - first_idx + 1, len(z_bin_centers), len(t_bin_centers)),
                compression=8,
                chunks=(1, len(z_bin_centers), len(t_bin_centers)),
                dtype='float32'
            )
            f_save.create_dataset(
                name='layer_histograms',
                shape=(last_idx - first_idx + 1, len(z_bin_layer_centers), len(t_bin_centers)),
                compression=8,
                chunks=(1, len(z_bin_layer_centers), len(t_bin_centers)),
                dtype='float32'
            )
            f_save.create_dataset(
                name='cell_kernels',
                shape=(last_idx - first_idx + 1, len(z_kernel), len(t_kernel)),
                compression=8,
                chunks=(1, len(z_kernel), len(t_kernel)),
                dtype='float32'
            )
            f_save.create_dataset(
                name='layer_kernels',
                shape=(last_idx - first_idx + 1, len(z_layer), len(t_kernel)),
                compression=8,
                chunks=(1, len(z_layer), len(t_kernel)),
                dtype='float32'
            )

            f_save.create_dataset(name='t_bin_centers', data=t_bin_centers, dtype='float32')
            f_save.create_dataset(name='t_bins', data=t_bins, dtype='float32')
            f_save.create_dataset(name='t_kernel', data=t_kernel, dtype='float32')
            f_save.create_dataset(name='z_bins', data=z_bins, dtype='float32')
            f_save.create_dataset(name='z_bins_layer', data=z_bins_layer, dtype='float32')
            f_save.create_dataset(name='z_bin_centers', data=z_bin_centers, dtype='float32')
            f_save.create_dataset(name='z_bin_layer_centers', data=z_bin_layer_centers, dtype='float32')
            f_save.create_dataset(name='z_kernel', data=z_kernel, dtype='float32')
            f_save.create_dataset(name='z_layer', data=z_layer, dtype='float32')
            f_save.create_dataset(name='z_layer_edges', data=z_layer_edges, dtype='float32')
            f_save.create_dataset(name='density', data=density, dtype='bool')
            f_save.attrs.create(name='z_step', data=Z_STEP)
            f_save.attrs.create(name='t_step', data=T_STEP)
            f_save.attrs.create(name='z_step_smooth', data=SMOOTH_Z_STEP)
            f_save.attrs.create(name='t_step_smooth', data=SMOOTH_T_STEP)
            f_save.attrs.create(name='z_units', data='um')
            f_save.attrs.create(name='t_units', data='ms')

        for chunk in range(first_chunk, last_chunk + 1):
            chunk_path = all_chunk_paths[chunk]
            print(f'Opening Chunk {chunk_path.stem}')

            with h5py.File(chunk_path, 'r') as f_chunk:
                chunk_iter_start = max(chunk_indices[chunk, 0], first_idx)
                chunk_iter_stop = min(chunk_indices[chunk, 1], last_idx) + 1
                for idx in range(chunk_iter_start, chunk_iter_stop):
                    idx_wrt_first = idx - first_idx
                    with h5py.File(cell_type_file, 'a') as f_save:
                        f_save['cell_histograms'][idx_wrt_first, :, :f_chunk[f'cell_histogram_{idx}'].shape[1]] = f_chunk[f'cell_histogram_{idx}'][:]
                        f_save['layer_histograms'][idx_wrt_first, :, :f_chunk[f'layer_histogram_{idx}'].shape[1]] = f_chunk[f'layer_histogram_{idx}'][:]
                        f_save['cell_kernels'][idx_wrt_first, :, :f_chunk[f'cell_smoothed_{idx}'].shape[1]] = f_chunk[f'cell_smoothed_{idx}'][:]
                        f_save['layer_kernels'][idx_wrt_first, :, :f_chunk[f'layer_smoothed_{idx}'].shape[1]] = f_chunk[f'layer_smoothed_{idx}'][:]


if __name__ == '__main__':
    # Working directory must be set to module
    param_path = MODULE_BASE.joinpath('reference_data/Miller_2025/axonal_delay_reduced_biphasic_params.hdf5')
    # ################################## Get Slurm Attributes ##########################################################
    parser = argparse.ArgumentParser(description='Runs Dendritic Delay Evaluations')
    parser.add_argument('-cdr', '--chunk_dir', help='Path to folder where results hdf5 files are to be saved', required=True)
    parser.add_argument('-dl', '--delays_dir', help='Path to folder containing axonal delay/z values', required=True)
    parser.add_argument('-kn', '--kernels_dir', help='Path to folder where final precomputed kernels are to be stored', required=True)

    args = parser.parse_args()
    chunk_dir = pathlib.Path(args.chunk_dir)
    delays_dir = pathlib.Path(args.delays_dir)
    kernels_dir = pathlib.Path(args.kernels_dir)

    ncpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    # ################################## Set User Parameters ###########################################################
    T_STEP = 0.1
    Z_STEP = 100
    SMOOTH_Z_STEP = 50  # 1 um
    SMOOTH_T_STEP = 0.01  # 0.025 ms
    DENSITY = True

    with h5py.File(param_path, 'r') as f_param:
        params = f_param['params'][:]

    layer_names = ['L23', 'L23_inh', 'L4', 'L4_inh', 'L5', 'L5_inh']

    params_extended = np.tile(params, (len(layer_names), 1))
    layer_names_extended = []
    for layer in layer_names:
        layer_names_extended = layer_names_extended + [layer] * params.shape[0]

    # nprocs = int(0.8 * multiprocessing.cpu_count())
    if ncpus > params_extended.shape[0]:
        ncpus = params_extended.shape[0]

    idx_chunks = np.array_split(np.arange(params_extended.shape[0]), ncpus)
    idx_chunks = [list(elm) for elm in idx_chunks]

    all_paths = sorted(list(chunk_dir.iterdir()))
    chunk_last_ids = np.array([int(pth.stem.split('_')[-1]) for pth in all_paths])
    chunks_complete = np.zeros(shape=(len(idx_chunks), ), dtype=bool)
    for i, file_path in enumerate(all_paths):
        with h5py.File(file_path, 'r') as f:
            last = int(f.attrs.__getitem__('last_index'))
            chunks_complete[i] = last >= chunk_last_ids[i]
    if np.all(chunks_complete):
        print('Chunks Already Complete, Performing Merge')
        merge_precomputed_kernel_chunks(
            chunk_folder=chunk_dir,
            all_params=params_extended,
            cell_types=layer_names_extended
        )
    else:
        print(f'Starting Computing\nOpening pool with {ncpus} Workers')
        with multiprocessing.Pool(processes=ncpus) as pool:

            partial_worker = partial(
                chunk_handler,
                parameters=params_extended,
                cell_type_labels=layer_names_extended
            )

            results = pool.map(
                partial_worker,
                idx_chunks,
            )
        print('All chunks complete :)')
        print('######################################')
        print('Merging')
        merge_precomputed_kernel_chunks(
            chunk_folder=chunk_dir,
            all_params=params_extended,
            cell_types=layer_names_extended
        )
        print('Merge Complete')

    print('fin :)')
