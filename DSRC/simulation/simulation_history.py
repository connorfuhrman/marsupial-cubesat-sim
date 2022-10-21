"""Dict which represents one timestep."""

from typing import TypedDict
import numpy as np


class SimulationHistoryTimestep(TypedDict):
    """A representation of one timestep in the simulation."""

    time: float
    """Simulation time."""
    craft_positions: dict[str, np.ndarray]
    """Mapping between Id and position at this timestep."""
    craft_types: dict[str, str]
    """Mapping between ID and string type."""


class SimulationHistoryMData(TypedDict):
    """Metadata about a simulation history."""

    max_num_crafts: int
    """The maximum number of crafts in the sim at one time."""
    total_iters: int
    """How many iterations were completed."""
    craft_ids: set[str]
    """All craft IDs that appeared in the simulation."""
    id: str
    """The ID of the simulation."""


class SimulationHistory(TypedDict):
    """Representation of a simulation history."""

    history: list[SimulationHistoryTimestep]
    metadata: SimulationHistoryMData
