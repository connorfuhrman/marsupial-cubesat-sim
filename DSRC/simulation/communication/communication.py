"""Model the act of spacecraft communication."""

from typing import Protocol
from dataclasses import dataclass
import logging
import numpy as np
from mpmath import mpf


class Message(Protocol):
    """Protocol of a message."""

    size: int
    """The size of the message in bytes."""

    def __eq__(self, o):
        """Require equality operator."""
        raise NotImplementedError()


class Spacecraft(Protocol):
    """Spacecraft protocol."""

    @property
    def position(self) -> np.ndarray:
        """Spacecraft has a position."""
        raise NotImplementedError()

    @property
    def id(self) -> int:  # noqa D
        raise NotImplementedError()

    def __eq__(self, o) -> bool:
        """Able the determine equality."""
        raise NotImplementedError()


@dataclass(init=False)
class Transmission:
    """A message being transmitted."""

    msg: Message
    """The message being send."""
    remaining_bytes: mpf
    """How many bytes left are there in the transmission."""

    def __init__(self, msg):  # noqa D
        self.msg = msg
        self.remaining_bytes = mpf(msg.size)

    def update(self, dt: mpf, crafts: dict[str, Spacecraft]) -> bool:
        """Do more of the transmission between sim steps.

        Returns True if the transmission is done.
        """
        if not self.is_valid(dt, crafts):
            return False
        # TODO define a datarate
        datrate = 10e6
        self.remaining_bytes -= datrate * dt
        return self.remaining_bytes <= 0.0

    def is_valid(self, dt: float, crafts: dict[str, Spacecraft]) -> bool:
        """Determine if the simulation is valid."""
        if self.msg.tx_id not in crafts or self.msg.rx_id not in crafts:
            # The sending or receiving craft isn't in the simulation
            return False
        return True

    @property
    def receiver(self) -> Spacecraft:  # noqa D
        return self.msg.msg["rx_id"]

    @property
    def sender(self) -> Spacecraft:  # noqa D
        return self.msg.msg["tx_id"]


class SimulationManager:
    """Class which manages the communication simulation.

    It's assumed that there are both 'active' and 'interrupted'
    transmissions. The interrupted ones are resumed if the comms
    link is reformed without any loss of data.
    """

    _logger: logging.Logger
    _transmissions: list[Transmission]

    def __init__(self, parentLogger: logging.Logger):
        """Initialize the comms sim."""
        self._logger = logging.getLogger(f"{parentLogger.name}.communication")
        self._transmissions = []

    def update(
        self, simtime: mpf, dt: mpf, crafts: dict[str, Spacecraft]
    ) -> dict[str, Spacecraft]:
        """Update one timestep."""
        remaining = []
        for t in self._transmissions:
            if t.update(dt, crafts):
                crafts[t.receiver].receive_msg(t.msg, simtime)
                self._logger.debug(
                    "Message from %s to %s delivered at %s",
                    t.sender,
                    t.receiver,
                    simtime,
                )
            else:
                remaining.append(t)
        self._transmissions = remaining
        if len(self._transmissions) > 0:
            self._logger.debug(
                "There are now %s active transmissions after update at %s.",
                len(self._transmissions),
                simtime,
            )
        return crafts

    def send_msg(self, msg: Message):
        """Start a transmission."""
        self._transmissions.append(Transmission(msg))
        self._logger.debug(
            "Queuing message send from %s to %s. "
            "There are now %s messages being send",
            msg.tx_id,
            msg.rx_id,
            len(self._transmissions),
        )
