"""Message definitions."""

import json
from typing import TypedDict
from dataclasses import dataclass
import numpy as np


class MessageData(TypedDict):
    """A generic message."""

    id: str
    timestamp: float


@dataclass(init=False)
class Message:
    """A message with data and size fields."""

    size: int
    msg: dict

    def __init__(self, msg):  # noqa D
        self.msg = msg
        self.size = len(bytes(json.dumps(self.msg)))


class SpacecraftState(MessageData):
    """The state of a spacecraft.

    Includes position, fuel, and if
    a sample has been caputed.
    """

    position: np.ndarray
    fuel_level: float
    has_sample: bool
