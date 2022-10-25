"""Message definitions.

These are messages which can be sent over a communication link.
Each message class holds the actual data and the size of the message which
is used to determine the sime-time delay between Tx and Rx.

Note that a message is an arbitrary dictionary but it's best practice
to define a message as a TypedDict (more specifically a child-class of the
MessageData class) for valid type hints during development and better support
from static anlysis tools.
"""

import json
from typing import TypedDict, Union
from dataclasses import dataclass
import numpy as np


class MessageData(TypedDict):
    """A generic message."""

    tx_id: str
    """ID of sender."""
    rx_id: str
    """ID of the receiver."""
    timestamp: float
    """Sim timestamp when Tx started."""


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def _json_dumps(obj):
    return json.dumps(obj, cls=_NumpyEncoder)


@dataclass(init=False)
class Message:
    """A message with data and size fields."""

    size: int
    """The size of the message in Bytes."""
    msg: MessageData
    """The message itself as a dict."""

    def __init__(self, msg: MessageData):  # noqa D
        self.msg = msg
        self.size = len(bytes(self.msg_str, encoding='utf8'))

    def __eq__(self, o) -> bool:  # noqa D
        return self.msg == o.msg

    @staticmethod
    def is_type(msg: Union["Message", MessageData], T) -> bool:
        """Is type(msg) T.

        True if all the expected keys exist.
        """
        if type(msg) is Message:
            msg = msg.msg
        expected_keys = np.array(T.__annotations__.keys(), dtype=str)
        msg_keys = np.array(msg.keys(), dtype=str)
        return np.array_equal(expected_keys, msg_keys)

    @property
    def tx_id(self) -> str:  # noqa D
        return self.msg['tx_id']

    @property
    def rx_id(self) -> str:  # noqa D
        return self.msg['rx_id']

    @property
    def timestamp(self) -> float:  # noqa D
        return self.msg['timestamp']

    @property
    def msg_str(self) -> str:
        """Encode message dict into a string."""
        return _json_dumps(self.msg)


class SpacecraftState(MessageData):
    """The state of a spacecraft.

    Includes position, fuel, and if
    a sample has been caputed.
    """

    position: np.ndarray
    fuel_level: float
    has_sample: bool


class SampleAquireCommand(MessageData):
    """Message to command a cubesat to aquire a given sample."""

    sample_pos: np.ndarray
