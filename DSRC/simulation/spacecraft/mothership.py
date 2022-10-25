"""Module to simulate a mothersip.

The mothership starts with some number of cubesats
which can be deployed and docked.
"""


from DSRC.simulation.spacecraft import Spacecraft, CubeSat
from DSRC.simulation.spacecraft import StraightLineAutopilot
from logging import Logger
import numpy as np


class Mothership(Spacecraft):
    """Class to represent a mothership."""

    _cubesat_capacity: int
    """How many cubesats can be docked at once."""
    _num_docked_cubesats: int
    """How many cubesats are currently docked."""

    def __init__(
        self,
        pos: np.ndarray,
        cubesat_capacity: int,
        parent_logger: Logger,
        fuel_level: float,
        vel: np.ndarray = None,
        rot_vel: np.ndarray = None,
        ori: np.ndarray = None,
    ):  # noqa D
        self._logger_name = "Mothership"
        super().__init__(
            pos, fuel_level, StraightLineAutopilot, parent_logger, vel, rot_vel, ori
        )
        self._cubesat_capacity = cubesat_capacity
        self._num_docked_cubesats = cubesat_capacity

    @property
    def can_deploy_cubesat(self) -> bool:
        """Can this mothership deploy a cubesat.

        Its assumed that the simulation which is maintain
        this mothership will instantiate a new cubesat in the
        simulation and manage that entity.
        """
        return self.num_docked_cubesats > 0

    @property
    def can_dock_cubesat(self) -> bool:
        """Can this mothership dock a cubesat."""
        return self.num_docked_cubesats < self.cubesat_capacity

    def deploy_cubesat(self) -> None:
        """Deploy the cubesat."""
        if not self.can_deploy_cubesat:
            raise RuntimeError("Cannot deploy any more cubesats")
        self._logger.debug("Deploying cubesat")
        self._num_docked_cubesats -= 1

    def dock_cubesat(self, craft: CubeSat) -> None:
        """Dock the cubesat."""
        self._logger.debug("Cubesat %s docked", craft.id)
        self._logger.info(
            "Cubesat docked with %s g of sample with value %s",
            craft.sample_weight,
            sum([s.value for s in craft.samples]),
        )
        self._num_docked_cubesats += 1

    @property
    def cubesat_capacity(self):  # noqa
        return self._cubesat_capacity

    @property
    def num_docked_cubesats(self):  # noqa D
        return self._num_docked_cubesats
