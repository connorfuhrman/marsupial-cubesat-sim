"""Utilities for spacecrafts to communicate."""

from dataclasses import dataclass
import numpy as np
from typing import Protocol


class Spacecraft(Protocol):
    """Spacecraft protocol."""

    @property
    def position(self) -> np.ndarray:
        """Spacecraft has a position."""
        raise NotImplementedError()

    def __eq__(self, o) -> bool:
        """Able the determine equality."""
        raise NotImplementedError()


@dataclass(eq=False)
class CommunicationLink:
    """A bidirectional comms link between two spacecrafts."""

    s1: Spacecraft
    s2: Spacecraft
    datarate: int = 10
    """The amount of data exchanged in some time [mB/s]"""

    def __eq__(self, o) -> bool:
        """Two links are equal if the same two ships are involved."""
        return (self.s1 == o.s1 or self.s1 == o.s2) and \
            (self.s2 == o.s2 or self.s2 == o.s1)

    def is_valid(self) -> bool:
        """Return a comms link if two spacecraft can communicate."""
        return np.linalg.norm(self.s1.position, self.s2.position) <= 5
