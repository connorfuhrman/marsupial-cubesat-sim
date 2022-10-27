logger_name = "Simulation"

from .simulation_history import (
    SimulationHistoryTimestep,
    SimulationHistory,
    SimulationHistoryMData,
)
from .animation import entrypoint as animate_simulation

from .simulation import Simulation, SimulationConfig, CubeSatConfig, MothershipConfig
