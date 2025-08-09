# noinspection PyUnresolvedReferences
import math

# noinspection PyUnresolvedReferences
import pathlib

# noinspection PyUnresolvedReferences
from neuron import h

# noinspection PyUnresolvedReferences
import neuron_interface.neurosim
from .cell_modification_parameters.cell_modification_parameters import (
    CellModificationParameters,
)
from .neuron_cell import NeuronCell


class NeuronCellTemplate(NeuronCell):

    @classmethod
    def from_id(
        cls,
        morphology_id,
        modification_parameters: CellModificationParameters = None,
        variation_seed=None,
    ):
        return cls(
            str(
                list(
                    pathlib.Path("morphology_path")
                    .joinpath(f"{morphology_id}")
                    .iterdir()
                )[0]
            ),
            morphology_id,
            modification_parameters,
            variation_seed,
        )

    @staticmethod
    def get_morphology_ids():
        return sorted(
            [
                int(f.name)
                for f in pathlib.Path("morphology_path").iterdir()
                if f.is_dir()
            ]
        )
