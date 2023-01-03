"""Message definitions.

These are messages which can be sent over a communication link.
Each message class holds the actual data and the size of the message which
is used to determine the sime-time delay between Tx and Rx.

Note that a message is an arbitrary dictionary but it's best practice
to define a message as a TypedDict (more specifically a child-class of the
MessageData class) for valid type hints during development and better support
from static anlysis tools.
"""

from typing import TypedDict, Union
from dataclasses import dataclass
import numpy as np
from DSRC.simulation.utils import to_json


class MessageData(TypedDict):
    """A generic message."""

    tx_id: str
    """ID of sender."""
    rx_id: str
    """ID of the receiver."""
    timestamp: float
    """Sim timestamp when Tx started."""


@dataclass(init=False)
class Message:
    """A message with data and size fields."""

    size: int
    """The size of the message in Bytes."""
    msg: MessageData
    """The message itself as a dict."""

    def __init__(self, msg: MessageData):  # noqa D
        self.msg = msg
        self.size = len(bytes(self.msg_str, encoding="utf8"))

    def __eq__(self, o) -> bool:  # noqa D
        return self.msg == o.msg

    @staticmethod
    def is_type(msg: Union["Message", MessageData], T) -> bool:
        """Is type(msg) T.

        True if all the expected keys exist.
        """
        if type(msg) is Message:
            msg = msg.msg
        expected_keys = set(T.__annotations__.keys())
        msg_keys = set(msg.keys())
        return msg_keys == expected_keys

    @property
    def tx_id(self) -> str:  # noqa D
        return self.msg["tx_id"]

    @property
    def rx_id(self) -> str:  # noqa D
        return self.msg["rx_id"]

    @property
    def timestamp(self) -> float:  # noqa D
        return self.msg["timestamp"]

    @property
    def msg_str(self) -> str:
        """Encode message dict into a string."""
        return to_json(self.msg)


####################################################################################################
# User-defined messages below.
#
# Each message must inherit from the MessageData class. The messages are all TypedDict's
# and their type is used to disbatch from a general queue of Message type objects.
# The Message.is_type method is used to determine if a dictionary is a certain message type.
# Since a TypedDict is just a generic dictionary at runtime to determine if a message is
# of a certain 'type' the dictionary's keys are compared to the expected keys from the
# class definition's annotations.
#
# E.g., one can do
#
# m = get_some_msg()
# dispatch = {
#     SampleAquireCommand: some_func,
#     SpacecraftState: another_func
# }
# for T, cb in dispatch.items():
#     if Message.is_type(m, T):
#         cb(m)
#         break
# raise TypeError
#
# Which tries to invoke a callback given a particular type.
####################################################################################################
class CubeSatState(MessageData):
    """The state of a cubesat.

    Includes position, fuel, and the value
    of the contained sample.
    """

    position: np.ndarray
    fuel_level: float
    sample_value: float


class SampleAquireCommand(MessageData):
    """Message to command a cubesat to aquire a given sample."""

    sample_pos: np.ndarray


class CubeSatDocked(MessageData):
    """Message sent from mothership that a cubesat has docked."""

    id: str
