import math
import numpy as np
from numpy.typing import NDArray
from ..neurosim.simulation.simulation import WaveformType
from ..neurosim.simulation.threshold_factor_simulation import ThresholdFactorSimulation
from ..spherical_cartesian_conversion import (
    norm_spherical_to_cartesian,
    spherical_to_cartesian,
)
from ..neurosim.cells.neuron_cell import NeuronCell


class SpikeRecordingRunner:
    """A runner that records action potentials for neurons under evenly spaced and grid sampled normalized e-field

    Parameters
    ----------
    waveform_type: WaveformType
        TMS waveform type object
    neurons: list[NeuronCell]
        List of NeuronCell objects to be simulated
    e_field_params : NDArray
        Electric field stimulation parameter sets defined by each row [theta, phi, relative_mag_change_per_mm, amplitude]
        for which all neurons will be simulated.
    simulation_length : float
        Total time for each simulation
    stimulation_delay: float
        Time at which stimulation starts
    simulation_time_step: float
        Time step for numerical simulation

    Attributes
    ----------
    waveform_type: WaveformType
        TMS waveform type object
    neurons: list[NeuronCell]
        List of NeuronCell objects to be simulated
    e_field_params : NDArray
        Electric field stimulation parameter sets defined by each row [theta, phi, relative_mag_change_per_mm, amplitude]
        for which all neurons will be simulated.
    simulation_length : float
        Total time for each simulation
    stimulation_delay: float
        Time at which stimulation starts
    simulation_time_step: float
        Time step for numerical simulation

    """

    def __init__(
        self,
        waveform_type: WaveformType,
        neurons: list[NeuronCell],
        e_field_params: NDArray,
        simulation_length: float,
        stimulation_delay: float,
        simulation_time_step: float,

    ):
        self.waveform_type = waveform_type
        self.neurons = neurons
        self.e_field_params = e_field_params
        self.simulation_length = simulation_length
        self.stimulation_delay = stimulation_delay
        self.simulation_time_step = simulation_time_step

    def run(self) -> list[list[dict]]:
        """Runs the action potential delay simulation for given e-field parameters, threshold, and neurons,
         and returns their segment ids and spike times.

        Returns
        -------
        list[list[dict]]
            Outer list with elements corresponding to each parameter set (row in e_field_params), each containing
            a list filled with dictionaries containing the results for each neuron. Dictionary keys are:
            'theta', 'phi', 'gradient', 'amplitude', 'stim_delay', 'segment_ids', and 'segment_spikes' for each simulation.

        """
        combined_results = []
        if self.e_field_params.ndim == 1:
            self.e_field_params = self.e_field_params[np.newaxis, :]

        for row_ind in range(np.shape(self.e_field_params)[0]):
            theta, phi, gradient, amplitude = self.e_field_params[row_ind, :]
            param_set_results = []
            for neuron in self.neurons:
                neuron_results = dict()
                neuron.load()
                simulation = ThresholdFactorSimulation(neuron_cell=neuron, waveform_type=self.waveform_type,
                                                       simulation_duration=self.simulation_length, stimulation_delay=self.stimulation_delay, simulation_time_step=self.simulation_time_step)
                simulation.attach()
                simulation.apply_parametric_e_field(theta, phi, gradient, 0, 0)
                seg_ids, seg_spikes = simulation.simulate(amplitude)
                simulation.detach()
                neuron.unload()
                neuron_results['theta'] = theta
                neuron_results['phi'] = phi
                neuron_results['gradient'] = gradient
                neuron_results['amplitude'] = amplitude
                neuron_results['stim_delay'] = simulation.stimulation_delay
                neuron_results['segment_ids'] = seg_ids
                neuron_results['segment_spikes'] = seg_spikes
                param_set_results.append(neuron_results)
            combined_results.append(param_set_results)
        return combined_results
