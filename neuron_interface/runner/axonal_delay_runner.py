import datetime
import numpy as np
import numpy.typing as npt
from typing import Type
import multiprocessing
from _functools import partial
import scipy.stats as stats
from neuron.units import ms
from numpy import ndarray
from ..neurosim.simulation.simulation import WaveformType
from ..neurosim.simulation.spike_simulation import SpikeSimulation
from ..spherical_cartesian_conversion import spherical_to_cartesian
from ..neurosim.cells import NeuronCell

THETA_MIN = 0.0
THETA_MAX = 180.0
PHI_MIN = 0.0
PHI_MAX = 360.0


class AxonalDelayRunner:
    """A axonal delay runner to evaluates the delays between TMS stimulation and axon terminal action potential arivals
    of neurons of a given layer under given electric field parameters, averaged across phi. Output of the simulations is
    a single timeseries representing the average output of the layer due to TMS, representated as a firing rate.

    Parameters
    ----------
    theta : float
        Polar angle of electric field in degrees
    relative_mag_change_per_mm : float
        The relative change of the E-field magnitude over the course of the somatodentritic axes of the neuron
        (Example: 25 -> 25 % E-field magnitude increase per mm over the course of the somatodentritic axes)
    intensity : float
        Electric field intensity in V/m
    phi_count : int
        Number of azimuthal angles
    waveform_type : WaveformType
        TMS coil waveform type (MONOPHASIC, BIPHASIC)
    stimulation_delay : float
        Delay before TMS onset in ms
    simulation_duration : float
        Length of simulations in ms
    simulation_time_step : float

    Attributes
    ----------
    theta : float
        Polar angle of electric field in degrees
    relative_mag_change_per_mm : float
        The relative change of the E-field magnitude over the course of the somatodentritic axes of the neuron
        (Example: 25 -> 25 % E-field magnitude increase per mm over the course of the somatodentritic axes)
    intensity : float
        Electric field intensity in V/m
    phi_count : int
        Number of azimuthal angles, defaulting to 60, steps of 3 degrees (always phi_count + 1 values, to include endpoint)
    cell_types: dict
        Dictionary of layer subtypes, each containing a list of string names of all electric types
    cell_constructors: dict
        Dictionary of layer subtypes, each containing a dictionary of electric types, each containing a list of
        references to a NeuronCell class or subclass for each morphology (order corresponds to cell_ids)
    cell_ids: dict
        Dictionary of layer subtypes, each containing a dictionary of electric types, each containing a list of string
        cell ids
    waveform_type : WaveformType
        TMS coil waveform type (MONOPHASIC, BIPHASIC)
    stimulation_delay : float
        Delay before TMS onset in ms
    simulation_duration : float
        Length of simulations in ms
    simulation_time_step : float
        Time step for simulations in ms
    e_field_directions_spherical : npt.NDArray[np.float_] shape=(phi_count * theta_count, 3)
        The spherical coordinates of the electric field directions to evaluate the threshold at
    e_field_directions_cartesian : npt.NDArray[np.float_] shape=(phi_count * theta_count, 3)
        The cartesian coordinates of the electric field directions to evaluate the threshold at
    """

    def __init__(
            self,
            theta: float,
            relative_mag_change_per_mm: float,
            intensity: float,
            cell_constructor: Type[NeuronCell],
            cell_ids: list,
            phi_count: int = 60,
            waveform_type: WaveformType = WaveformType.BIPHASIC,
            stimulation_delay: float = 0.0 * ms,
            simulation_duration: float = 2.0 * ms,
            simulation_time_step: float = 0.005 * ms
    ):
        assert (theta >= THETA_MIN) & (theta <= THETA_MAX), 'theta must be between 0 and 180 degrees.'
        assert intensity >= 0, 'intensity must be non-negative.'
        assert phi_count > 0, 'phi_count must be nonzero.'
        assert stimulation_delay >= 0.0, 'stimulation_delay must be non-negative.'
        assert simulation_duration > 0.0, 'simulation_duration must be greater than zero'
        assert (simulation_time_step > 0.0) & (simulation_time_step < simulation_duration), 'simulation time step must be greater than zero and less than the simulation length'

        self.theta: float = theta
        self.relative_mag_change_per_mm: float = relative_mag_change_per_mm
        self.intensity: float = intensity
        self.cell_constructor = cell_constructor
        self.cell_ids = cell_ids
        self.phi_count: int = phi_count
        self.waveform_type: WaveformType = waveform_type
        self.stimulation_delay = stimulation_delay * ms
        self.simulation_duration = simulation_duration * ms
        self.simulation_time_step = simulation_time_step * ms

        self.e_field_directions_spherical = np.array(
            np.meshgrid(
                np.linspace(PHI_MIN, PHI_MAX, phi_count + 1)[:-1],
                np.full((phi_count, ), theta),
                1,
            )
        ).T.reshape(-1, 3)
        self.e_field_directions_cartesian = spherical_to_cartesian(
            self.e_field_directions_spherical
        )

    def run(
        self,
        processes: int = 1,
    ) -> tuple[list, ndarray]:
        """Runs the axon TMS stimulation simulation for every angle phi for all cell morphologies in this layer.
        Returns a list of results for each parameter set, containing action potential arrival times, z position of each
        compartment with respect to the cell soma.

        Parameters
        ----------
        processes : int, optional
            The number of processes to use to evaluate the thresholds, by default 1

        Returns
        -------
        cell_electric_type_results : dict[dict[list[list[float], list[float]]]]
            Dictionary of layer subtypes, each containing a dictionary of electric types, each containing a list
            [[delays], [z values]] of action potential delays w.r.t TMS onset, and z positions of each compartment
            with respect to cell soma.
        simulation_results : list[list[list[int], list[float]]]
            List of results for each parameter set before combining phis, in lists as
            [[segment_ids], [action_potential_times]]
        all_sim_parameters : npt.ArrayLike
            Array containing all simulation parameters before combining phis with columns
            [theta, gradient, intensity, phi, cell_constructor, cell_ids]
        """
        # Generate simulation full parameter set
        phi_range = np.linspace(PHI_MIN, PHI_MAX, self.phi_count + 1)[:-1]
        neuron_cells = []
        all_sim_parameters = []
        for cell_id in self.cell_ids:
            # Construct Cell and organize params by cell object
            neuron_cells.append(self.cell_constructor.from_id(cell_id))
            for phi in phi_range:
                all_sim_parameters.append(
                    [self.theta,
                     self.relative_mag_change_per_mm,
                     self.intensity,
                     phi,
                     int(cell_id)]
                )
        neuron_cells = np.array(neuron_cells)
        all_sim_parameters = np.row_stack(all_sim_parameters)

        # Simulate with multiprocessing
        if processes > 1:
            param_tuples = zip(np.array_split(neuron_cells, processes))

            with multiprocessing.Pool(processes) as pool:
                partial_worker = partial(
                    AxonalDelayRunner._simulate_action_potentials,
                    phis=phi_range,
                    theta=self.theta,
                    relative_mag_change_per_mm=self.relative_mag_change_per_mm,
                    intensity=self.intensity,
                    waveform_type=self.waveform_type,
                    simulation_duration=self.simulation_duration,
                    stimulation_delay=self.stimulation_delay,
                    simulation_time_step=self.simulation_time_step
                )
                chunked_results = pool.starmap(
                    partial_worker,
                    param_tuples
                )
            simulation_results = []
            for chunk_result_list in chunked_results:
                for cell_results in chunk_result_list:
                    simulation_results.append(cell_results)
        # Simulate serially
        else:
            simulation_results = AxonalDelayRunner._simulate_action_potentials(
                neuron_cells=neuron_cells,
                phis=phi_range,
                theta=self.theta,
                relative_mag_change_per_mm=self.relative_mag_change_per_mm,
                intensity=self.intensity,
                waveform_type=self.waveform_type,
                simulation_duration=self.simulation_duration,
                stimulation_delay=self.stimulation_delay,
                simulation_time_step=self.simulation_time_step
                )
        print("All cells complete")
        # Convert action potential arrivals to delay at terminals, and determine terminal z positions
        # Organize delay results by cell electric type

        cell_result_lengths = []
        for cell_results in simulation_results:
            for param_result in cell_results:
                cell_result_lengths.append(len(param_result[0]))  # get length of cell ids for each param set

        # Combine delays for different phi, with the same cell type
        # for simulation_result, phi, cell_constructor, cell_id, cell_subtype, cell_electric_type in iterable:
        all_simulation_results = []
        if sum(cell_result_lengths) != 0:
            for i, (neuron_cell, cell_results) in enumerate(zip(neuron_cells, simulation_results)):
                if cell_result_lengths[i] != 0:
                    neuron_cell.load()
                    cell_type_name = neuron_cell.__class__.__name__
                    cell_subtype = cell_type_name.split('_')[1]
                    cell_electric_type = cell_type_name.split('_')[2]
                    for phi_index in range(len(phi_range)):
                        compartment_ids, action_potential_times = cell_results[phi_index]
                        # If any action potentials were recorded calculate delay for axon terminal segments
                        if len(action_potential_times) > 0:
                            delays, z, terminal_ids = AxonalDelayRunner._extract_terminal_delay(
                                cell=neuron_cell,
                                segment_ids=compartment_ids,
                                action_potentials=action_potential_times,
                                stimulation_delay=self.stimulation_delay
                            )
                            all_simulation_results.append([delays, z, terminal_ids])
                        else:
                            all_simulation_results.append([[], [], []])
                    neuron_cell.unload()
                else:
                    for _ in range(len(phi_range)):
                        all_simulation_results.append([[], [], []])

        return all_simulation_results, all_sim_parameters

    @staticmethod
    def _simulate_action_potentials(
            neuron_cells: npt.NDArray[NeuronCell],
            phis: npt.NDArray[np.float_],
            theta: float,
            relative_mag_change_per_mm: float,
            intensity: float,
            waveform_type: WaveformType,
            simulation_duration: float,
            stimulation_delay: float,
            simulation_time_step: float
    ) -> list[list[list[list[int], list[float]]]]:
        """
        Simulates the given list of cells at all values of phi, for the fixed theta, gradient, intensity, waveform,
        duration, stimulation delay, and time step. Results are organized by cell.
        Parameters
        ----------
        neuron_cells : npt.NDArray[NeuronCell]
            Unloaded neuron cells to simulate
        phis : npt.ArrayLike[float]
            Phi values to simulate in order
        theta: float
            Theta polar angle for all simulations
        relative_mag_change_per_mm: float
            Gradient in E field for all simulations
        intensity: float
            E field intensity for all simulations
        waveform_type: WaveformType
            TMS waveform type for all simulations
        simulation_duration: float
            Simulation length for all simulations
        stimulation_delay: float
            Delay before stimulation onset for all simulations
        simulation_time_step: float
            Time step for all simulations

        Returns
        -------
        list[list[list[int], list[float]]]
            Returns list of results for each cell, containing a list for each phi result containing
            [[segment_ids], [action_potential_times]]
        """
        simulation_results = []
        for index, neuron_cell in enumerate(neuron_cells):
            cell_results = []
            neuron_cell.load()
            simulation = SpikeSimulation(
                neuron_cell=neuron_cell,
                waveform_type=waveform_type,
                stimulation_delay=stimulation_delay,
                simulation_time_step=simulation_time_step,
                simulation_duration=simulation_duration
            )
            time = datetime.datetime.now()
            time_str = time.strftime('%d/%m/%Y %H:%M:%S')
            print(f'Simulating cell: {neuron_cell.__class__.__name__} id: {neuron_cell.morphology_id} | {index + 1}/{len(neuron_cells)} | {time_str}')
            for phi in phis:
                # Apply E Field
                simulation.apply_parametric_e_field(
                    e_field_theta=theta,
                    e_field_phi=phi,
                    relative_mag_change_per_mm=relative_mag_change_per_mm,
                    phi_change_per_mm=0,
                    theta_change_per_mm=0
                )
                # Attach Simulation and Spike Recording
                simulation.attach()
                # Simulate and detach
                # ids, action_potentials = simulation.simulate_interrupted(amplitude_scaling_factor=self.intensity)
                ids, action_potentials = simulation.simulate(amplitude_scaling_factor=intensity)
                simulation.detach()
                cell_results.append([ids, action_potentials])
            neuron_cell.unload()
            simulation_results.append(cell_results)
            tdelta = datetime.datetime.now() - time
            print(f'Completed cell: {neuron_cell.__class__.__name__} id: {neuron_cell.morphology_id} | Iteration time: {tdelta.seconds}s')
        return simulation_results

    @staticmethod
    def _extract_terminal_delay(
            cell: Type[NeuronCell],
            segment_ids: list[int],
            action_potentials: list[float],
            stimulation_delay: float
    ) -> tuple[list[float], list[float], list[int]]:
        """
        Locates axon terminal segments in provided cell and returns the combined spike delays with respect to TMS onset
        given action potential times.

        Parameters
        ----------
        cell : NeuronCell
            NeuronCell object
        segment_ids : list[int]
            List of compartment ids for the given cell (tree ordered)
        action_potentials : list[float]
            List of action potential times for given cell (ordered corresponding to segment_ids)
        stimulation_delay : float
            Delay between simulation start and TMS stimulation onset in ms

        Returns
        -------
        tuple
            list[float]
                List of delays at axon terminals (may not include all terminals in cell)
            list[float]
                List of z position of axon terminals (corresponding to terminals in delays)
            list[int]
                List of axon terminal ids (corresponding to terminals in delays)

        Raises
        ______
        ValueError
            If the given NeuronCell is not loaded
        """
        assert cell.loaded, "Neuron cell is not loaded"

        # determine indexes of axon terminals for this cell
        terminal_indexes = []
        terminal_z = []
        terminal_spike_delays = []
        soma_z = cell.soma[0](0.5).z_xtra
        ind = 0
        for section in cell.all:
            for seg_ind, segment in enumerate(section):
                if (section in cell.axon) & (len(section.children()) == 0) & (
                        seg_ind == len(list(section)) - 1):
                    # If global segment index in result segment indices
                    if ind in segment_ids:
                        segment_index_in_results = np.where(np.array(segment_ids) == ind)[0][0]
                        terminal_spike_delays.append(action_potentials[segment_index_in_results])
                        # Subtract action potential time from stimulation onset
                        terminal_spike_delays[-1] -= stimulation_delay
                        # Append terminal index and z for this delay
                        terminal_indexes.append(ind)
                        terminal_z.append(segment.z_xtra - soma_z)
                ind += 1

        return terminal_spike_delays, terminal_z, terminal_indexes
