from typing import Type

from neuron import h
from ..cells import NeuronCell
from .simulation import Simulation, WaveformType
from neuron.units import ms, mV


class SpikeSimulation(Simulation):
    """A NEURON simulation with the functionality to run simulations and record action potentials in all compartments
    for the applied e-field. Optional simulation which prematurely exits simulations with no action potentials.
    """

    def __init__(
        self,
        neuron_cell: Type[NeuronCell],
        waveform_type: WaveformType,
        stimulation_delay=0.0 * ms,
        simulation_time_step=0.005 * ms,
        simulation_duration=2.0 * ms,
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

    def attach(self):
        """Attaches spike recording to the neuron and connects the simulation initialization method to the global NEURON space."""
        self._init_spike_recording()
        super().attach()

    def detach(self):
        """Removes the spike recording from the neuron and disconnects the initialization method."""
        for net in self.netcons:
            net.record()
        self.netcons.clear()
        super().detach()

    def _init_spike_recording(self):
        """Initializes spike recording for every segment of the neuron."""
        self.netcons = []
        i = 0
        for section in self.neuron_cell.all:
            for segment in section:
                recording_netcon = h.NetCon(segment._ref_v, None, sec=section)
                recording_netcon.threshold = 0 * mV
                recording_netcon.delay = 0
                recording_netcon.record(
                    self._action_potentials, self._action_potentials_recording_ids, i
                )
                self.netcons.append(recording_netcon)
                i += 1

    def simulate(
            self,
            amplitude_scaling_factor: float
    ) -> tuple[list[int], list[float]]:
        """Wrapper for super.simulate() with spike recording. Runs a single simulation for
        given electric field intensity, returns two lists containing the ids of all segments
        and spike recording times at those segments.

        Parameters
        __________
        amplitude_scaling_factor : float
            The scaling factor for the amplitude of the stimulation waveform

        Returns
        _______
        list[int]
            Segment ids for all segments in morphology tree order
        list[float]
            Spike times for all segments corresponding to those in the id list

        Raises
        ______
        ValueError
            If the simulation is not attached to any neuron
        ValueError
            If the neuron cell is not loaded
        """

        if not self.attached:
            raise ValueError("Simulation is not attached")
        if not self.neuron_cell.loaded:
            raise ValueError("Neuron cell is not loaded")

        super().simulate(amplitude_scaling_factor)

        return (
            list(int(idval) for idval in list(self._action_potentials_recording_ids)),
            list(self._action_potentials)
        )
