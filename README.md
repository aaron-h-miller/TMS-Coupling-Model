# Miller 2025 TMS Coupling
### Authors
- Aaron Miller, miller@cbs.mpg.de
- Konstantin Weise, kweise@cbs.mpg.de (main contact)
- Torge Worbs, torwo@dtu.dk (Root code of NEURON interface adapted with permission from [^fn1])

## Cloning Repo
```
cd existing_repo
git remote add origin https://gitlab.gwdg.de/aaron.miller/miller-2025-tms-coupling.git
git branch -M main
git push -uf origin main
```

## Build Repository

Setup a virtual environment with the provided environment_pip.yaml, or environment_forge.yaml with the following command
```
conda env create -f environment_<distro>.yaml
conda activate coupling_model
```

Compiling Neuron Mechanisms (.mod files). Generates x86_64 folder containing C compiled NEURON mechanisms.
```
cd neuron_interface/neurosim/mechanisms
nrnivmodl
```

## Download Reference Data
In order for many scripts to function, reference data folder called `Miller_2025` from the OSF directory DOI: 10.17605/OSF.IO/NXVDH (https://osf.io/nxvdh/) must be
downloaded and dropped directly as a subfolder of `reference_data`.

## Repository Structure

### batch_runners
Files which execute simulations in large multiprocessed batches, meant to be executed via slurm. 
(Data merging scripts are included)

### figure_plotting
Example scripts which simulate (or recall simulated data) and plot the published figures. Additional plotting scripts 
are provided to recall data from the dendritic delay lookup table and to plot DI-waves.

### helper_functions
A series of libraries and helper functions are grouped by use and type and are used across the module.

### neuron_interface
The core simulator of the module, which includes cell morphologies, simulation classes, and runners that execute
simulations. 

#### neurosim
##### cells
Contains `NeuronCell` subclasses for the studied cell subtypes. cells_hoc contains the cell morphologies, organized by 
cell index (the cell indices used in this work are logged in the library `helper_functions/cell_type_lib.py`, dataclass
attribute `CellType.cell_ids`). Cell morphologies are recovered from [^fn2] with adaptations consistent with [^fn1] and [^fn3].
##### coil_recordings
Contains TMS coil recordings for Mono/Biphasic TMS.
##### mechanisms
Contains `.mod` mechanism files that NEURON uses to determine point process and distributed synaptic process dynamics. 
It is necessary to run the NEURON commandline function `nrnivmodl` in this directory to compile the .mod files to C, which 
generates a `x86_64` folder and tells NEURON where to find the mechanisms.   
##### simulation
Simulation classes that serve various purposes to simulate dynamics on axonal arbors and dendritic trees. Simulation 
instances are used in the runner files as well as directly in the example plotting scrips in `figure_plotting`.

#### runner
Contains runner scripts which deploy simulation instances and are used in `figure_plotting` and `batch_runners` scripts.

### postprocessing_modules
Classes which are helpful for postprocessing raw simulation data. Particularly 'AxonalDelayKernel' which processes 
axonal delay simulation results (delay and cortical depth values) to produce spike density distributions that are 
directly used as inputs to dendritic delay simulations and for visualization 
(see `neuron_interface/neurosim/simulation/dendritic_delay_simulation.py` and 
`figure_plotting/axonal_delay_directional_sensitiviy_Fig8_9_10_FigS2_7.py`).

### reference_data
Comes with reference data used from other research works, generated parameter files used in this work, and simulation 
results from this work. The following folders and files are contained in `Miller_2025` must be downloaded from the OSF repository DOI: 10.17605/OSF.IO/NXVDH (https://osf.io/nxvdh/) and dropped directly
into `reference_data` in order for many scripts to function as expected:
`Miller_2025` should become a direct child of `reference_data`, and contains:

- `delay_z_data`: Contains `.hdf5` files with axonal delay and z (cortical depth) values precalculated for L23, L4, L5 cells
- `axon_kernels_precomputed'`: Contains precomputed AxonalDelayKernel objects archived into `.hdf5` files.
- `dendritic_current`: Contains dendritic current gPC results (`.hdf5` and `.pkl`) as well as an extensive lookup table of dendritic currents at various electric field inputs for free gPC parameters chosen at their average values.

## References

[^fn1]: Konstantin Weise, Torge Worbs, Benjamin Kalloch, Victor H. Souza, Aurélien Tristan Jaquier, Werner Van Geit, Axel Thielscher, Thomas R. Knösche; Directional sensitivity of cortical neurons towards TMS-induced electric fields. Imaging Neuroscience 2023; 1 1–22. doi: https://doi.org/10.1162/imag_a_00036

[^fn2]: Markram, H., Muller, E., Ramaswamy, S., Reimann, M. W., Abdellah, M., Sanchez, C. A., Ailamaki, A., Alonso-Nanclares, L., Antille, N., Arsever, S., Kahou, G. A., Berger, T. K., Bilgili, A., Buncic, N., Chalimourda, A., Chindemi, G., Courcol, J. D., Delalondre, F., Delattre, V., Druckmann, S., … Schürmann, F. (2015). Reconstruction and Simulation of Neocortical Microcircuitry. Cell, 163(2), 456–492. https://doi.org/10.1016/j.cell.2015.09.029

[^fn3]: Aberra, A. S., Wang, B., Grill, W. M., & Peterchev, A. V. (2020). Simulation of transcranial magnetic stimulation in head model with morphologically-realistic cortical neurons. Brain stimulation, 13(1), 175–189. https://doi.org/10.1016/j.brs.2019.10.002