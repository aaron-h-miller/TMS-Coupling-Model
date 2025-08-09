import math
import pathlib
from enum import Enum
from typing import Type

import numpy as np
import numpy.typing as npt
from neuron import h
from scipy.io import loadmat
from neuron.units import ms, mV

import neuron_interface.neurosim

from ...spherical_cartesian_conversion import norm_spherical_to_cartesian
from ..cells.neuron_cell import NeuronCell


class WaveformType(Enum):
    MONOPHASIC = 1
    BIPHASIC = 2


class Simulation:
    """Wrapper to set up, modify and execute a NEURON simulation of a single cell.

    Parameters
    --------------------------
    neuron_cell : NeuronCell
        The neuron that is supposed to be used in the NEURON simulation.
    waveform_type : WaveformType
        The waveform type that is supposed to be used in the NEURON simulation


    Attributes
    ----------------------------------
    neuron_cell: NeuronCell
        The cell that is supposed to be simulated
    stimulation_delay: float
        Initial delay before the activation waveform is applied in s
    simulation_temperature: float
        Temperature for the simulation in degree Celsius
    simulation_time_step: float
        The time step used for the simulation in ms
    simulation_duration: float
        The duration of the simulation in ms
    waveform: list[float]
        The amplitude values of the waveform used
    waveform_time:list[float]
        The time values of the waveform used
    """

    INITIAL_VOLTAGE = -70 * mV

    def __init__(
        self,
        neuron_cell: Type[NeuronCell],
        waveform_type: WaveformType,
        stimulation_delay=0.005 * ms,
        simulation_time_step=0.005 * ms,
        simulation_duration=1.0 * ms,
    ):
        self.neuron_cell = neuron_cell

        self.stimulation_delay = stimulation_delay
        self.simulation_temperature = 37
        self.simulation_time_step = simulation_time_step
        self.simulation_duration = simulation_duration
        self.waveform, self.waveform_time, self.waveform_duration = self._load_waveform(waveform_type)

        self.init_handler = None
        self.init_state = None
        self.attached = False

    def attach(self):
        """Attaches spike recording to the neuron and connects the simulation initialization method to the global NEURON space."""
        self.init_handler = h.FInitializeHandler(2, self._post_finitialize)

        self.attached = True

    def detach(self):
        """Removes the spike recording from the neuron and disconnects the initialization method."""
        del self.init_handler
        self.init_handler = None

        self.attached = False

    def _post_finitialize(self):
        """Initialization method to unsure a steady state before the actual simulation is started."""
        temp_dt = h.dt

        h.t = -1e11 * ms
        h.dt = 1e9 * ms

        while h.t < -h.dt:
            h.fadvance()

        h.dt = temp_dt
        h.t = 0 * ms
        h.fcurrent()
        h.frecord_init()

    def simulate(self, amplitude_scaling_factor: float) -> None:
        """Executes a NEURON simulation with the submitted amplitude as the scaling factor for the E-field

        Parameters
        ----------
        stimulation_amplitude : float
            Then scaling factor for the amplitude of the stimulation waveform
        """
        if self.init_state is None:
            h.celsius = self.simulation_temperature
            h.dt = self.simulation_time_step
            h.tstop = self.simulation_duration
            h.finitialize(self.INITIAL_VOLTAGE)
            self.init_state = h.SaveState()
            self.init_state.save()
        else:
            self.init_state.restore()

        waveform_vector = h.Vector(self.waveform * amplitude_scaling_factor)
        waveform_time_vector = h.Vector(self.waveform_time)

        waveform_vector.play(h._ref_stim_xtra, waveform_time_vector, 1)

        h.run()

    def _load_waveform(self, waveform_type: WaveformType):
        """Loads the submitted waveform and modifies it to fit the simulation settings."""
        tms_waves = loadmat(
            str(
                pathlib.Path(neuron_interface.neurosim.__file__)
                .parent.joinpath("coil_recordings/TMSwaves.mat")
                .absolute()
            )
        )

        recorded_time = tms_waves["tm"].ravel()
        recorded_e_field_magnitude = tms_waves["Erec_m"]

        if waveform_type is WaveformType.BIPHASIC:
            recorded_e_field_magnitude = tms_waves["Erec_b"]

        sample_factor = int(self.simulation_time_step / np.mean(np.diff(recorded_time)))
        if sample_factor < 1:
            sample_factor = 1

        simulation_time = recorded_time[::sample_factor]
        simulation_e_field_magnitude = np.append(
            recorded_e_field_magnitude[::sample_factor], 0
        )

        waveform_duration = simulation_time[-1]

        if self.stimulation_delay >= self.simulation_time_step:
            pre_time = np.arange(0, self.stimulation_delay, self.simulation_time_step)
            simulation_time = np.concatenate(
                (
                    pre_time,
                    simulation_time + self.stimulation_delay,
                )
            )
            simulation_e_field_magnitude = np.concatenate(
                (
                    np.zeros_like(pre_time),
                    np.append(recorded_e_field_magnitude[::sample_factor], 0),
                )
            )

        simulation_time = np.append(
            np.concatenate(
                (
                    simulation_time,
                    np.arange(
                        simulation_time[-1] + self.simulation_time_step,
                        self.simulation_duration,
                        self.simulation_time_step,
                    ),
                )
            ),
            self.simulation_duration,
        )

        if len(simulation_time) > len(simulation_e_field_magnitude):
            simulation_e_field_magnitude = np.pad(
                simulation_e_field_magnitude,
                (0, len(simulation_time) - len(simulation_e_field_magnitude)),
                constant_values=(0, 0),
            )
        else:
            simulation_e_field_magnitude = simulation_e_field_magnitude[
                : len(simulation_time)
            ]

        return simulation_e_field_magnitude, simulation_time, waveform_duration

    def apply_e_field(self, e_field_at_segments: npt.ArrayLike):
        """Applies the submitted E-field vectors at each segment by calculating the quasi potentials from the E-field vectors

        Parameters
        ----------
        e_field_at_segments : npt.ArrayLike (N x 3)
            E-field vectors for each segment ordered by section index (inside all) and segment index (inside section)
        """
        self.init_state = None
        e_field_at_segments = np.array(e_field_at_segments)
        i = 0
        for section in self.neuron_cell.all:
            for segment in section:
                segment.Ex_xtra, segment.Ey_xtra, segment.Ez_xtra = e_field_at_segments[
                    i
                ]
                i += 1

        current_section = self.neuron_cell.soma[0]
        current_section.es_xtra = 0
        section_stack = list(current_section.children())
        parent_stack = [current_section(1)] * len(section_stack)

        while len(section_stack) > 0:
            current_section = section_stack.pop()
            segments = list(current_section)
            parent_segment = parent_stack.pop()

            for segment in segments:
                segment.es_xtra = parent_segment.es_xtra - 0.5 * 1e-3 * (
                    (parent_segment.Ex_xtra + segment.Ex_xtra)
                    * (segment.x_xtra - parent_segment.x_xtra)
                    + (parent_segment.Ey_xtra + segment.Ey_xtra)
                    * (segment.y_xtra - parent_segment.y_xtra)
                    + (parent_segment.Ez_xtra + segment.Ez_xtra)
                    * (segment.z_xtra - parent_segment.z_xtra)
                )
                parent_segment = segment

            children = list(current_section.children())
            section_stack += children
            parent_stack += [parent_segment] * len(children)

    def apply_parametric_e_field(
        self,
        e_field_theta: float,
        e_field_phi: float,
        relative_mag_change_per_mm: float,
        phi_change_per_mm: float,
        theta_change_per_mm: float,
    ) -> None:
        """Applies an artificial parameter based E-field at each segment by calculating the quasi potentials from the E-field vectors.

        Parameters
        ----------
        e_field_theta : float
            The polar angle of the E-field at the soma in relation to the somatodentritic axes of the neuron
        e_field_phi : float
            The azimuthal angle of the E-field at the soma in relation to the somatodentritic axes of the neuron
        relative_mag_change_per_mm : float
            The relative change of the E-field magnitude over the course of the somatodentritic axes of the neuron
        phi_change_per_mm : float
            The relative change of the E-field azimuthal angle over the course of the somatodentritic axes of the neuron
        theta_change_per_mm : float
            The relative change of the E-field polar angle over the course of the somatodentritic axes of the neuron
        """

        cell_segment_coordinates = self.neuron_cell.get_segment_coordinates()

        soma_position_z = self.neuron_cell.soma[0](0.5).z_xtra
        min_z = np.amin(cell_segment_coordinates[:, 2])
        max_z = np.amax(cell_segment_coordinates[:, 2])
        soma_distance_to_min = math.fabs(soma_position_z - min_z)
        soma_distance_to_max = math.fabs(soma_position_z - max_z)

        magnitude_at_max = 1 + soma_distance_to_max / 1000 * (
            (-relative_mag_change_per_mm) / 100
        )
        magnitude_at_min = 1 + soma_distance_to_min / 1000 * (
            relative_mag_change_per_mm / 100
        )

        phi_at_max = soma_distance_to_max / 1000 * phi_change_per_mm
        phi_at_min = soma_distance_to_min / 1000 * (-phi_change_per_mm)

        theta_at_max = soma_distance_to_max / 1000 * theta_change_per_mm
        theta_at_min = soma_distance_to_min / 1000 * (-theta_change_per_mm)

        e_field_at_segments = []
        i = 0
        for section in self.neuron_cell.all:
            for _ in section:
                gradient_phi = np.interp(
                    cell_segment_coordinates[i][2],
                    [min_z, soma_position_z, max_z],
                    [phi_at_min, 0, phi_at_max],
                )
                phi = e_field_phi + gradient_phi

                gradient_theta = np.interp(
                    cell_segment_coordinates[i][2],
                    [min_z, soma_position_z, max_z],
                    [theta_at_min, 0, theta_at_max],
                )
                theta = e_field_theta + gradient_theta
                e_field_direction = norm_spherical_to_cartesian([phi, theta])

                gradient_magnitude = max(
                    np.interp(
                        cell_segment_coordinates[i][2],
                        [min_z, soma_position_z, max_z],
                        [magnitude_at_min, 1, magnitude_at_max],
                    ),
                    0,
                )
                e_field_at_segment = e_field_direction * gradient_magnitude
                e_field_at_segments.append(e_field_at_segment)
                i += 1
        self.apply_e_field(e_field_at_segments)
