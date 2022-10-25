"""Module to simulate a cubesat.

Cubesat class is a child of the Spacecraft class.
"""

from DSRC.simulation.spacecraft import Spacecraft, StraightLineAutopilot
import numpy as np
from scipy.stats import bernoulli
from typing import Protocol
from logging import Logger


class Sample(Protocol):
    """A protocol which represents a sample.

    This class defines a duck-typed interface
    which represents a sample. The cubesat
    intends to capture the sample.
    """

    weight: float
    value: float


class CubeSat(Spacecraft):
    """A CubeSat for sample return."""

    _samples: list[Sample]
    """Hold the captured sample if one was collected."""
    _is_deployed: bool
    """Is this cubesat deployed or docked?"""
    _dist: bernoulli
    """Bernoilli probability distribution."""

    def __init__(
        self,
        loc: np.ndarray,
        fuel_capacity: float,
        sample_prob: float,
        parentLogger: Logger,
        *,
        is_deployed: bool = True,
        vel: np.ndarray = None,
        rot_vel: np.ndarray = None,
        ori: np.ndarray = None
    ):
        """Initialize the cubesat.

        Initialization can either have the cubesat deployed
        or not. If it's not deployed it's assumed to be docked
        in the mothership
        """
        self._logger_name = "CubeSat"
        super().__init__(
            loc, fuel_capacity, StraightLineAutopilot, parentLogger, vel, rot_vel, ori
        )
        self._is_deployed = is_deployed
        self._has_sample = False
        self._samples = []
        self._dist = bernoulli(sample_prob)

    def attempt_sample_capture(self, sample: Sample) -> bool:
        """Attempt a sample capture.

        This function simply samples from a Bernoilli
        distribution and returns that boolean value
        which indicates if the cubesat was able to
        capture the sample or not.

        If the cubesat was able to capture a sample
        it will be holding the sample's weight
        """
        if not self.is_deployed:
            raise RuntimeError("Cannot attempt sample capture when not deployed.")
        captured = bool(self._dist.rvs())
        if captured:
            self._samples.append(sample)
            self._logger.debug(
                "Captured the sample! This craft now holds %s grams of sample",
                self.sample_weight,
            )
        else:
            self._logger.debug("Failed to capture sample :(")
        return captured

    @property
    def has_sample(self) -> bool:  # noqa D
        return len(self.samples) > 0

    @property
    def samples(self) -> list[Sample]:  # noqa D
        return self._samples

    @property
    def sample_weight(self) -> float:
        """The weight of all collected samples."""
        return sum([s.weight for s in self.samples])

    @property
    def is_deployed(self) -> bool:  # noqa D
        return self._is_deployed
