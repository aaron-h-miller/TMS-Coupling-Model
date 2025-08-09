from typing import Type

import numpy as np
from neuron import h

from ..cells.neuron_cell import NeuronCell
from .simulation import WaveformType
from .spike_simulation import SpikeSimulation
from neuron.units import ms, mV


class ThresholdFactorSimulation(SpikeSimulation):
    """A NEURON simulation with the functionality to run simulations and to find a threshold scaling factor for
    the applied e-field that is the minimum factor to trigger an action potential.
    """

    def __init__(
        self,
        neuron_cell: Type[NeuronCell],
        waveform_type: WaveformType,
        stimulation_delay=0.005 * ms,
        simulation_time_step=0.005 * ms,
        simulation_duration=1.0 * ms,
    ):
        super().__init__(
            neuron_cell,
            waveform_type,
            stimulation_delay,
            simulation_time_step,
            simulation_duration,
        )
        self.netcons = []
        self._action_potentials = h.Vector()
        self._action_potentials_recording_ids = h.Vector()

    def find_threshold_factor(self) -> tuple[float, list[int], list[float]]:
        """Searches for the minimal threshold factor to trigger an action potential in the simulated neuron.

        Returns
        -------
        float
            The threshold factor that was found

        Raises
        ------
        ValueError
            If the simulation is not attached to any neuron
        ValueError
            If the neuron cell is not loaded
        """
        if not self.attached:
            raise ValueError("Simulation is not attached")
        if not self.neuron_cell.loaded:
            raise ValueError("Neuron cell is not loaded")
        low = 0
        high = 1e6
        amplitude = 100
        epsilon = 1e-8 + 5e-2

        while low <= 0 or high >= 1e6:
            self.simulate(amplitude)
            if self._action_potentials.size() >= 3:
                high = amplitude
                amplitude = amplitude / 2
            else:
                low = amplitude
                amplitude = amplitude * 2
            if low > high:
                return np.inf, [], []
            if high < 0.000001:
                return 0, [], []

        amplitude = (high + low) / 2

        while high - low > epsilon:
            self.simulate(amplitude)
            if self._action_potentials.size() >= 3:
                high = amplitude
            else:
                low = amplitude
            amplitude = (high + low) / 2
        return (
            high,
            list(self._action_potentials_recording_ids),
            list(self._action_potentials),
        )



