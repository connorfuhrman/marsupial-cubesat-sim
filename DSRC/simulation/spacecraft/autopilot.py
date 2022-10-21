"""An 'autopilot' system.

This holds modules for a spacecraft 'autopiliot', i.e.,
something which calculates the lower-level velocity
commands.
"""

import numpy as np
from collections import deque
from logging import Logger, getLogger


class StraightLineAutopilot:
    """Simple autopilot.

    The autopilot will set the velocity in the
    direction of the next waypoint to capture.
    """

    _waypoints: deque[np.ndarray] = None
    """Waypoints to follow."""
    _tracking_waypnt: np.ndarray
    """The waypoint we're headed to."""
    _waypoint_capture_tol: float = 0.05
    """Distance away from a waypoint when it's considered captured [m]."""
    _heading: np.ndarray
    """The current heading command."""
    _logger: Logger
    """Logging facilities."""

    def __init__(self, parent_logger: Logger):  # noqa D
        self._logger = getLogger(f"{parent_logger.name}.autopilot")
        self._heading = np.empty(3)
        self._tracking_waypnt = None
        self._waypoints = deque()

    def add_waypoint(self, pnt: np.ndarray, front: bool = False) -> None:
        """Add a waypoint to the path."""
        # if self._tracking_waypnt is not None and np.linalg.norm(pnt - self._tracking_waypnt) < 0.01:
        #     self._logger.debug("Commanded waypoint of %s is the same as the tracked waypoint. "
        #                        "It will be disreguarded.", pnt)
        #     return
        if front:
            # if self._tracking_waypnt is not None:
            #     self._waypoints.appendleft(self._tracking_waypnt)  # Save what was tracked
            self._tracking_waypnt = pnt  # Now this point is being tracked
        else:
            self._waypoints.append(pnt)
        self._logger.debug(f"Added {pnt} to list of waypoints. "
                           f"There are now {self.num_waypoints} waypoints.")

    def clear_waypoints(self) -> None:
        """Clear the waypoints."""
        self._logger.debug("Clearing waypoints")
        while self.num_waypoints > 0:
            _ = self._waypoints.pop()
        self._tracking_waypnt = None

    def update(self, pos: np.ndarray) -> np.ndarray:
        """Calculate the velocity needed to reach the next waypoint."""
        # Copy to ensure no invalid references
        if self.num_waypoints == 0 and self._tracking_waypnt is None:
            self._heading = np.zeros(3, dtype=float)
        elif self._tracking_waypnt is None and self.num_waypoints > 0:
            self._tracking_waypnt = self._waypoints.popleft()
            self._logger.debug("There is not a waypoint being tracked. "
                               "Tracking %s as the next waypoint. "
                               "There are %s more waypoints in the sequence.",
                               self._tracking_waypnt, self.num_waypoints)
            self._calc_heading(pos)
        elif self._waypoint_captured(pos):
            # Only if we've captured the tracking waypoint
            # should we calculate a new velocity vector
            self._logger.debug(f"Captured waypint at {self._tracking_waypnt}")
            if self.num_waypoints == 0:
                self._logger.debug("No more waypoints")
                self._heading = np.zeros(3, dtype=float)
                self._tracking_waypnt = None
            else:
                self._tracking_waypnt = self._waypoints.popleft()
                self._calc_heading(pos)
                self._logger.debug("New heading is %s tracking waypoing %s", self._heading, self._tracking_waypnt)
        else:
            self._calc_heading(pos)
        return self._heading

    def _waypoint_captured(self, pos: np.ndarray) -> bool:
        """Has the current waypoint been captured."""
        return np.linalg.norm(self._tracking_waypnt - pos) < self._waypoint_capture_tol

    def _calc_heading(self, pos: np.ndarray) -> None:
        """Calculate the new heading."""
        self._heading = self._tracking_waypnt - pos
        self._heading /= np.linalg.norm(self._heading)

    @property
    def num_waypoints(self) -> int:  # noqa D
        return len(self._waypoints)

    @property
    def curr_waypoint(self) -> np.ndarray:  # noqa D
        return self._tracking_waypnt.copy() if self._tracking_waypnt is not None else None
