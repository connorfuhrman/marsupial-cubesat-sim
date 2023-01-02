"""Module to simulate a generic spacecraft."""

import numpy as np
import logging
from shortuuid import uuid
from copy import copy
from typing import Union, Protocol, Tuple
from queue import Queue
from DSRC.simulation.communication import CommsSimManager, Message
from DSRC.simulation.communication import SpacecraftStateMsg


class Autopilot(Protocol):
    """Autopilot module.

    The autopilot system should navigate the craft
    to a given waypoint. This abstracts from the user
    the need to calculate heading commands manually.

    The update function returns a unit-vector
    heading which is used to calculate a velocity command.
    """

    def __init__(self, logger_name: str):  # noqa D
        raise NotImplementedError()

    def add_waypoint(self, waypoint: np.ndarray):
        """Set a waypoint in 3-space.

        This sets the waypoint where the spacecraft
        should navigate. On the subsequent updates.
        """
        raise NotImplementedError()

    def clear_waypoints(self) -> None:
        """Reset and optionally stop."""

    def update(self, pos: np.ndarray) -> np.ndarray:
        """Get the next heading command."""
        raise NotImplementedError()


class Spacecraft:
    """A Spacecraft model."""

    _position: np.ndarray
    """The 3-space location of the spacecraft in [x,y,z] in [m]"""
    _orientation: np.ndarray = np.array([0, 0, 0])
    """Array holding [roll, pitch, yaw] of craft in [rad]"""
    _velocity: np.ndarray = np.array([0, 0, 0])
    """3-space velocity of spacecraft in [x,y,z] in [m]"""
    _rotational_velocity: np.ndarray
    """Rotational velocity in [roll, pitch, yaw] in [rad/s]"""
    _autopilot: Autopilot
    """Autopilot module."""
    _fuel_capacity: float
    """How much fuel can this craft hold."""
    _fuel_level: float
    """Remaining fuel in arbitrary units."""
    _msg_queue: Queue[Tuple[float, Message]]
    """Received messages since the last timestep."""
    _logger: logging.Logger
    """Logging functionality."""
    _logger_name: str
    """Name for the logger."""
    _id: str
    """Unique ID for this spacecraft."""

    def __init__(
        self,
        loc: np.ndarray,
        fuel_level: float,
        autopilotClass: Autopilot,
        parentLogger: logging.Logger,
        vel: np.ndarray = None,
        rot_vel: np.ndarray = None,
        ori: np.ndarray = None,
        vel_mag: float = 1.0,
    ):
        """Initialize a spacecraft.

        Initialize with a location and, optionally,
        a velocity and orientation.
        """
        self._id = uuid()
        self._logger_name = "Spacecraft"
        self._fuel_level = fuel_level
        self._fuel_capacity = fuel_level
        self._msg_queue = Queue()

        def assign_arr(val, attr, dtype=float, sz: int = 3):
            val = np.array(val, dtype=dtype)
            self._check_arr_sz(val, sz)
            self.__setattr__(attr, val)

        default = lambda: np.array([0, 0, 0])  # noqa E731

        assign_arr(loc, "_position")
        assign_arr(default() if vel is None else vel, "_velocity")
        assign_arr(default() if rot_vel is None else rot_vel, "_rotational_velocity")
        assign_arr(default() if ori is None else ori, "_orientation")
        self.vel_mag = vel_mag  # Can be set in an experiment if need to throttle

        self._logger = logging.getLogger(
            f"{parentLogger.name}." f"spacecraft.{self.id}"
        )
        self._autopilot = autopilotClass(self._logger)
        self._logger.debug(
            "Constructed at position %s " "with velocity %s " "and orientation %s",
            self.position,
            self.velocity,
            self.orientation,
        )

    def update_kinematics(self, dtime, vel_mag: float = None) -> bool:
        """Update the kinematic state of the craft.

        The autopilot member is queried for the heading
        and the craft applies velocity in that heading with
        a magnitude of vel_mag which deaults to 1 m/s

        If the function returns false then the craft has
        run out of fuel and is now dead!.
        """
        dt = float(dtime)  # In case this is passed as mpf
        heading = self._autopilot.update(self.position)
        if vel_mag is None:
            vel_mag = self.vel_mag
        self._velocity = vel_mag * heading
        self._position += dt * self.velocity
        self._orientation += dt * self._rotational_velocity
        # TODO Update fuel level based on acceleration or other kinematic attribute
        if self.fuel_level is None:  # Unlimited fuel
            return True
        else:
            self._fuel_level -= np.random.uniform(low=0.0, high=0.25)
            return self.fuel_level > 0.0

    def apply_rot_vel(self, rot_vel: np.ndarray) -> None:
        """Apply some rotational velocity."""
        self._rotational_velocity += rot_vel

    def add_waypoint(self, pnt: np.ndarray, front: bool = False) -> None:
        """Add a waypoint to the autopilot."""
        self._autopilot.add_waypoint(pnt, front)

    def clear_waypoints(self, stationary: bool = False) -> None:
        """Instruct the autopilot to clear all waypoints."""
        self._autopilot.clear_waypoints()

    def drop_curr_waypoint(self) -> None:
        """Stop tracking the current waypoint."""
        self._autopilot.drop_curr_waypoint()

    def receive_msg(self, msg: Message, timestamp: float):
        """Get a message from another craft."""
        self._msg_queue.put((timestamp, msg))

    def get_msg(self) -> Union[Tuple[float, Message], None]:
        """Get the top of the queue or None if there's not msgs."""
        if self.msg_queue_size == 0:
            return None
        else:
            return self._msg_queue.get()

    def get_state_msg(
        self, time: float, as_msg_obj: bool = False
    ) -> Union[SpacecraftStateMsg, Message]:
        """Get a state message for this spacecraft."""
        return self._get_msg(
            SpacecraftStateMsg(
                tx_id=self.id,
                timestamp=time,
                position=self.position,
                fuel_level=self.fuel_level,
                has_sample=self.has_sample,
            ),
            as_msg_obj,
        )

    def _get_msg(
        self, msg: dict, as_msg_obj: bool
    ) -> Union[dict, Message]:  # noqa D 400
        """Convenince function:

        Get a msg either as the dict of a Message obj.
        """
        if as_msg_obj:
            return Message(msg)
        else:
            return msg

    def _check_arr_sz(self, arr: np.array, sz: int = 3) -> None:
        if len(arr) != sz:
            raise ValueError("Invalid array size")

    @property
    def position(self) -> np.ndarray:
        """Retrieve the current 3-space position."""
        self._check_arr_sz(self._position)
        return self._position.copy()

    @property
    def orientation(self) -> np.ndarray:
        """Retrieve the current orientation."""
        self._check_arr_sz(self._orientation)
        return self._orientation.copy()

    @property
    def velocity(self) -> np.ndarray:
        """Retrive the current velocity."""
        self._check_arr_sz(self._velocity)
        return self._velocity.copy()

    @property
    def rotational_velocity(self) -> np.ndarray:
        """Retrive the current rotational_velocity."""
        self._check_arr_sz(self._rotational_velocity)
        return self._rotational_velocity.copy()

    @property
    def fuel_level(self) -> float:
        """Retrieve fuel level."""
        return self._fuel_level

    @property
    def fuel_capacity(self) -> float:
        """Retrieve the fuel capacity."""
        return self._fuel_capacity

    @property
    def num_waypoints(self) -> int:
        """Return the crafts waypoints."""
        return self._autopilot.num_waypoints

    @property
    def curr_waypoint(self) -> np.ndarray:
        """Return where we're currently navigating."""
        return self._autopilot.curr_waypoint

    @property
    def msg_queue_size(self) -> int:  # noqa D401
        """The number of messages awaiting dispatch."""
        return self._msg_queue.qsize()

    @property
    def has_msg(self) -> bool:  # noqa D
        return self.msg_queue_size > 0

    @property
    def id(self) -> str:
        """Retrieve a copy of this craft's ID."""
        return copy(self._id)

    @property
    def logger(self) -> logging.Logger:  # noqa D
        return self._logger
