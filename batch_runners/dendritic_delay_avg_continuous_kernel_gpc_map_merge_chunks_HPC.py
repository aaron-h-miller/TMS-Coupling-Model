
import argparse
import os
import pathlib
import time
import datetime
import h5py
import numpy as np
from enum import Enum
from helper_functions.multiprocessing_chunks_complete import all_chunks_complete
from helper_functions.load_cell_distributions import compute_distribution
from helper_functions import __file__
MODULE_BASE = pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    # ###################################### Retrieve run arguments ####################################################
    parser = argparse.ArgumentParser(description='Runs Dendritic Delay Evaluations')
    parser.add_argument('-sv', '--save_path', help='Path to folder where results hdf5 files are to be saved',
                        required=True)
    parser.add_argument('-pm', '--param_path', help='Path to parameter file hdf5', required=True)
    parser.add_argument('-ch', '--chunk_path', help='Path to folder where chunk hdf5 files are stored',
                        required=True)
    args = parser.parse_args()

    # User provided parameters
    param_path = pathlib.Path(args.param_path)
    seed = int(param_path.stem.split('_')[-3])
    num_samples = int(param_path.stem.split('_')[-1])
    save_path = pathlib.Path(args.save_path)
    chunk_folder = pathlib.Path(args.chunk_path)
    save_file = save_path.joinpath(f'merged_dendritic_delay_current_{seed}_{num_samples}.hdf5')
    save_file_raw = save_path.joinpath(f'merged_raw_dendritic_delay_current_{seed}_{num_samples}.hdf5')
    # ################################ Load parameter file #############################################################
    print('Load Parameter File')
    t1 = time.perf_counter()
    with h5py.File(param_path, 'r') as f_param:
        params = f_param['params'][:].astype('float64')
        cols = f_param['params'].attrs.__getitem__('columns')
        unique_free_params = f_param['free_params'][:].astype('float64')
        free_index_mask = f_param['free_index_mask'][:].astype('int64')
    col_enum = Enum('Columns', [col_label for col_label in cols], start=0)
    t2 = time.perf_counter()
    print(f'Parameter File Loaded: {datetime.timedelta(seconds=t2-t1)}')
    # ################################ Verify all chunks are complete ##################################################
    if not save_file.is_file():
        t1 = time.perf_counter()
        print('Checking if Chunks are all Complete')
        all_complete, complete_list = all_chunks_complete(chunk_folder=chunk_folder)
        if not all_complete:
            chunks_remaining = np.where(np.invert(complete_list))[0]
            raise AssertionError(f'Chunks are not complete, remaining chunks are: {chunks_remaining}')
        t2 = time.perf_counter()
        print(f'Chunks Complete, Collecting Files: {datetime.timedelta(seconds=t2-t1)}')
    # ################################ Collect Chunk Files and Parameters ##############################################
    glob_list = list(chunk_folder.glob('*chunk_[0-9]*_[0-9]*.hdf5'))
    sorted_chunk_files = sorted(glob_list, key=lambda pth: int(pth.stem.split('_')[-2]))
    chunk_index_bounds = np.vstack(
        [[int(pth.stem.split('_')[-2]), int(pth.stem.split('_')[-1])] for pth in sorted_chunk_files]
    )
    with h5py.File(sorted_chunk_files[0], 'r') as f_first:
        time_axis = f_first['t'][:]
        # sim_duration = float(f_first.attrs.__getitem__('simulation_duration'))
        # sim_time_step = float(f_first.attrs.__getitem__('simulation_time_step'))

    cell_ids = params[:, col_enum['cell_id'].value]
    unique_cell_ids = np.unique(cell_ids)
    soma_depths = params[:, col_enum['soma_depth'].value]
    unique_soma_depths = np.unique(soma_depths)[::-1]  # ordered from close to CSF toward White Matter
    l5_cell_density, _, _, z_mean, _, _ = compute_distribution(
        cell_class='L5',
        z_discrete_samples=unique_soma_depths
    )
    # Ensure weight function for average is normalized to sum = 1 across the provided samples
    density_sample_weight_function = l5_cell_density / np.sum(l5_cell_density)
    # ############################### Iterate Chunks and Average #######################################################
    if not save_file.is_file():
        # Initialize Merged File
        print('Initializing Merged Files')
        t1 = time.perf_counter()
        with h5py.File(save_file, 'w') as f_save:
            f_save.create_dataset(name='t', data=time_axis, dtype='float32')
            # f_save.create_dataset(name='simulation_duration', data=sim_duration, dtype='float32')
            # f_save.create_dataset(name='simulation_time_step', data=sim_time_step, dtype='float32')
            f_save.create_dataset(
                name='current',
                shape=(unique_free_params.shape[0], len(time_axis)),
                compression=8,
                chunks=(1, len(time_axis))
            )
            raw_pars = f_save.create_dataset(name='raw_params', data=params, dtype='float32')
            f_save['raw_params'].attrs.create(name='columns', data=cols)
            pars = f_save.create_dataset(name='params', data=unique_free_params, dtype='float32', compression=8)
            raw_pars.attrs.create(name='columns', data=cols)
            pars.attrs.create(name='columns', data=cols[:6])

        with h5py.File(save_file_raw, 'w') as f_save_raw:
            f_save_raw.create_dataset(name='t', data=time_axis, dtype='float32')
            # f_save.create_dataset(name='simulation_duration', data=sim_duration, dtype='float32')
            # f_save.create_dataset(name='simulation_time_step', data=sim_time_step, dtype='float32')
            f_save_raw.create_dataset(
                name='current',
                shape=(params.shape[0], len(time_axis)),
                compression=8,
                chunks=(1, len(time_axis))
            )
            f_save_raw.create_dataset(name='raw_params', data=params, dtype='int32')
            f_save_raw['raw_params'].attrs.create(name='columns', data=cols)
        start_idx = 0
        t2 = time.perf_counter()
        print(f'Beginning Merge: {datetime.timedelta(seconds=t2-t1)}')
    else:
        with h5py.File(save_file, 'r') as f:
            last_idx = int(f.attrs.get('last_index'))
        start_idx = last_idx + 1
        if start_idx >= unique_free_params.shape[0]:
            print('Merge Already Complete')
        else:
            print(f'Merge Resuming from free parameters {start_idx} | {unique_free_params[start_idx, :]}')
    for free_param_counter in range(start_idx, unique_free_params.shape[0]):
        free_par_set = unique_free_params[free_param_counter, :]
        t1 = time.perf_counter()
        parameter_set_mask = free_index_mask == free_param_counter
        # Check parameter set mask
        parameter_set_mask_alternate = np.all(params[:, :6] == free_par_set, axis=1)
        if not np.all(parameter_set_mask == parameter_set_mask_alternate):
            print(f'Ids from mask: {parameter_set_mask}')
            print(f'Ids from np.all compare: {parameter_set_mask_alternate}')
            raise IndexError('Free parameter mask does not correspond to all simulation params')
        current_per_cell = np.zeros(shape=(len(unique_cell_ids), len(time_axis)))
        # Average across cell morphologies
        for cell_id_counter, cell_id in enumerate(unique_cell_ids):
            cell_id_mask = cell_ids == cell_id
            soma_depth_currents = np.zeros(shape=(len(unique_soma_depths), len(time_axis)))
            # Weighted average across soma depths
            for soma_depth_counter, soma_depth in enumerate(unique_soma_depths):
                soma_depth_mask = soma_depths == soma_depth
                # Get combined mask for specific free parameters, cell_id, and soma depth
                combined_mask = parameter_set_mask & cell_id_mask & soma_depth_mask
                param_index = np.where(combined_mask)[0][0]
                # Determine chunk file where this index is contained
                chunk_index = np.where(
                    (param_index >= chunk_index_bounds[:, 0]) & (param_index <= chunk_index_bounds[:, 1])
                )[0][0]
                with h5py.File(sorted_chunk_files[chunk_index], 'r') as f_chunk:
                    current = f_chunk[f'current_{param_index}'][:].astype('float64')
                    with h5py.File(save_file_raw, 'a') as f_raw:
                        f_raw['current'][param_index, :] = current
                soma_depth_currents[soma_depth_counter, :] = current
            # avg across soma depths weighted by cell density
            depth_avg_current = np.average(soma_depth_currents, axis=0, weights=density_sample_weight_function)
            current_per_cell[cell_id_counter, :] = depth_avg_current
        free_params_avg_current = np.mean(current_per_cell, axis=0)
        with h5py.File(save_file, 'a') as f_save:
            f_save['current'][free_param_counter, :] = free_params_avg_current
            f_save.attrs.create(name='last_index', data=free_param_counter, dtype='int32')
        t2 = time.perf_counter()
        print(f'Complete Idx {free_param_counter}/{len(unique_free_params) - 1} Params: {free_par_set} | {t2 - t1:.1}s')
    print('Merge Complete, fin :)')

