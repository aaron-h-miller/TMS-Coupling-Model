from typing import Type

import numpy as np
from neuron import h

from ..cells.neuron_cell import NeuronCell
from .simulation import Simulation, WaveformType
from neuron.units import ms, mV


class VoltageCurveSimulation(Simulation):
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
        self.time_recording = h.Vector()
        self.voltage_recordings = []

    def attach(self):
        """Attaches voltage recording to the neuron and connects the simulation initialization method to the global NEURON space."""
        self.time_recording.record(h._ref_t)
        for section in self.neuron_cell.all:
            for segment in section:
                self.voltage_recordings.append(h.Vector())
                self.voltage_recordings[-1].record(segment._ref_v)
        super().attach()

    def detach(self):
        """Removes voltage recording from the neuron and disconnects the initialization method."""
        self.time_recording.record()
        self.time_recording = h.Vector()

        for voltage_recording in self.voltage_recordings:
            voltage_recording.record()

        self.voltage_recordings = []

        super().detach()

    def get_voltage_curves(self, factor):
        if not self.attached:
            raise ValueError("Simulation is not attached")
        if not self.neuron_cell.loaded:
            raise ValueError("Neuron cell is not loaded")

        self.simulate(factor)

        voltage_curves = []
        for voltage_recording in self.voltage_recordings:
            voltage_curves.append(list(voltage_recording))

        return np.array(list(self.time_recording)), np.array(voltage_curves)
