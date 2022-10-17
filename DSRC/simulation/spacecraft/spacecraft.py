"""Module to simulate a generic spacecraft."""

import numpy as np
import logging
from shortuuid import uuid
from copy import copy


class Spacecraft:
    """A Spacecraft model."""

    _position: np.array
    """The 3-space location of the spacecraft in [x,y,z] in [m]"""
    _orientation: np.array = np.array([0, 0, 0])
    """Array holding [roll, pitch, yaw] of craft in [rad]"""
    _velocity: np.array = np.array([0, 0, 0])
    """3-space velocity of spacecraft in [x,y,z] in [m]"""
    _logger: logging.Logger
    """Logging functionality."""
    _id: str
    """Uninique ID for this spacecraft."""

    def __init__(self,
                 loc: np.array,
                 vel: np.array = None,
                 ori: np.array = None):
        """Initialize a spacecraft.

        Initialize with a location and, optionally,
        a velocity and orientation.
        """
        loc = np.array(loc, dtype=float)
        self._check_arr_sz(loc)
        self._position = loc
        self._id = uuid()

        if vel is not None:
            vel = np.array(vel, dtype=float)
            self._check_arr_sz(vel)
            self._velocity = vel

        if ori is not None:
            ori = np.array(ori, dtype=float)
            self._check_arr_sz(ori)
            self._orientation = ori

        self._logger = logging.getLogger(f"DSRC.Simulation.Spacecraft.{self.id}")
        self._logger.debug("Constructed at position %s "
                           "with velocity %s "
                           "and orientation %s",
                           self.position, self.velocity, self.orientation)

    def update_kinematics(self, dt: float) -> None:
        """Update the craft's kinematic state.

        Updates the position based on the current velocity
        for a timestep :param:dt

        TODO: Update velocity based on loss model (?) and update
              orientation based on rotational velocity.
        """
        self._position += (dt * self._velocity)
        #self._velocity *= 0.98

    def _check_arr_sz(self, arr: np.array, sz: int = 3) -> bool:
        if len(arr) != sz:
            raise ValueError("Invalid array size")

    @property
    def position(self) -> np.array:
        """Retrieve the current 3-space position."""
        self._check_arr_sz(self._position)
        return copy(self._position)

    @property
    def orientation(self) -> np.array:
        """Retrieve the current orientation."""
        self._check_arr_sz(self._orientation)
        return copy(self._orientation)

    @property
    def velocity(self) -> np.array:
        """Retrive the current velocity."""
        self._check_arr_sz(self._velocity)
        return copy(self._velocity)

    @property
    def id(self) -> str:
        """Retrieve a copy of this craft's ID."""
        return copy(self._id)
