"""Dict which represents one timestep."""

from typing import TypedDict
import numpy as np
from mpmath import mpf


class SimulationHistoryTimestep(TypedDict):
    """A representation of one timestep in the simulation."""

    time: mpf
    """Simulation time."""
    craft_positions: dict[str, np.ndarray]
    """Mapping between Id and position at this timestep."""
    craft_types: dict[str, str]
    """Mapping between ID and string type."""
    sample_positions: list[np.ndarray]
    """All samples in the simulation at this time."""


class SimulationHistoryMData(TypedDict):
    """Metadata about a simulation history."""

    max_num_crafts: int
    """The maximum number of crafts in the sim at one time."""
    max_num_samples: int
    """The maximum number of samples over all iters."""
    craft_ids: set[str]
    """All craft IDs that appeared in the simulation."""
    id: str
    """The ID of the simulation."""
    total_iters: int
    """How many iterations were completed."""
    time_start: float
    """Epoch time (seconds) at sim start.

    Specifically this is at sim construction time.
    """
    time_end: float
    """Epoch time (seconds) at sim end.

    Specifically, this is right before returning.
    """
    sim_time: mpf
    """Total simulated seconds."""


class SimulationHistory(TypedDict):
    """Representation of a simulation history."""

    history: list[SimulationHistoryTimestep]
    metadata: SimulationHistoryMData
