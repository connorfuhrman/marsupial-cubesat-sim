"""Model the act of spacecraft communication."""

from DSRC.simulation.communication import CommunicationLink
from typing import Protocol
from dataclasses import dataclass


class Message(Protocol):
    """Protocol of a message."""

    size: int
    """The size of the message in bytes."""

    def __eq__(self, o):
        """Require equality operator."""
        raise NotImplementedError()


@dataclass(init=False)
class Transmission:
    """A message being transmitted."""

    msg: Message
    """The message being send."""
    remaining_bytes: float
    """How many bytes left are there in the transmission."""
    link: CommunicationLink
    """The comm link used for this transmission."""

    def __init__(self, msg, link):  # noqa D
        self.msg = msg
        self.remaining_bytes = msg.size
        self.link = link

    def update(self, dt: float) -> bool:
        """Do more of the transmission between sim steps.

        Returns True if the transmission is done.
        """
        self.remaining_bytes -= float(self.link.datarate) * dt
        return self.remaining_bytes <= 0.0


class SimulationManager:
    """Class which manages the communication simulation."""

    _active_transmissions: set[Transmission]
    _interrupted_transmissions: set[Transmission]

    def __init__(self):
        """Initialize the comms sim."""
        self._active_transmissions = set()
        self._interrupted_transmissions = set()

    def _check_for_reformed_links(self):
        for t in self._interrupted_transmissions:
            if t.link.is_valid():
                self._active_transmissions.add(t)
                self._interrupted_transmissions.remove(t)

    def _check_links(self):
        self._check_for_reformed_links()
        for t in self._active_transmissions:
            if not t.link.is_valid():
                self._active_transmissions.remove(t)
                self._interrupted_transmissions.add(t)

    def update(self, dt):
        """Update one timestep."""
        self._check_links()
        for t in self._active_transmissions:
            if t.update(dt):
                t.receiver.receive_msg(t.link, t.msg)
                self._active_transmissions.remove(t)

    def send_msg(self, msg: Message, rx, tx):
        """Start a transmission."""
        link = CommunicationLink(rx, tx)
        self._active_transmissions.add(Transmission(msg, link))

    @property
    def valid_links(self) -> list[CommunicationLink]:
        """Get a deep copy list of all valid links now in the sim."""
        self._check_links()
        return [t.link for t in self._active_transmissions]

    @property
    def num_valid_links(self) -> int:
        """Get the number of valid links."""
        self._check_links()
        return len(self._active_transmissions)
