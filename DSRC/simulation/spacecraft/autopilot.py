"""An 'autopilot' system.

This holds modules for a spacecraft 'autopiliot', i.e.,
something which calculates the lower-level velocity
commands.
"""

import numpy as np
from queue import Queue
from logging import Logger, getLogger


class StraightLineAutopilot:
    """Simple autopilot.

    The autopilot will set the velocity in the
    direction of the next waypoint to capture.
    """

    _waypoints: Queue[np.ndarray] = Queue()
    """Waypoints to follow."""
    _tracking_waypnt: np.ndarray = None
    """The waypoint we're headed to."""
    _waypoint_capture_tol: float = 1
    """Distance away from a waypoint when it's considered captured [m]."""
    _heading: np.ndarray = np.empty(3)
    """The current heading command."""
    _logger: Logger
    """Logging facilities."""

    def __init__(self, parent_logger_name: str):  # noqa D
        self._logger = getLogger(f"{parent_logger_name}.autopilot")

    def add_waypoint(self, pnt: np.ndarray) -> None:
        """Add a waypoint to the path."""
        self._waypoints.put(pnt)
        self._logger.debug(f"Added {pnt} to list of waypoints. "
                           f"There are now {self._waypoints.qsize()} waypoints.")

    def clear_waypoints(self) -> None:
        """Clear the waypoints."""
        self._logger.debug("Clearing waypoints")
        while self._waypoints.qsize() > 0:
            _ = self._waypoints.get()
        self._tracking_waypnt = None

    def update(self, pos: np.ndarray) -> np.ndarray:
        """Calculate the velocity needed to reach the next waypoint."""
        if self._waypoints.qsize() == 0:
            self._heading = np.array([0, 0, 0], dtype=float)
        elif self._tracking_waypnt is None:
            self._tracking_waypnt = self._waypoints.get()
            self._logger.info("There is not a waypoint being tracked. "
                               "Tracking %s as the next waypoint. "
                               "There are %s more waypoints in the sequence.",
                               self._tracking_waypnt, self._waypoints.qsize())
            self._calc_heading(pos)
        elif self._waypoint_captured(pos):
            # Only if we've captured the tracking waypoint
            # should we calculate a new velocity vector
            self._logger.debug(f"Captured waypint at {self._tracking_waypnt}")
            self._tracking_waypnt = self._waypoints.get()
            if self._waypoints.empty():
                self._logger.info("Current waypoint is the last in the sequence")
            self._calc_heading(pos)
            self._logger.debug(f"New heading is {self._heading}")
        return self._heading

    def _waypoint_captured(self, pos: np.ndarray) -> bool:
        """Has the current waypoint been captured."""
        return np.linalg.norm(self._tracking_waypnt - pos) < self._waypoint_capture_tol

    def _calc_heading(self, pos: np.ndarray) -> None:
        """Calculate the new heading."""
        self._heading = self._tracking_waypnt - pos
        self._heading /= np.linalg.norm(self._heading)
