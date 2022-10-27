"""Simulation of Decentralized Sample Recovery Coordination."""


from DSRC.simulation.spacecraft import Spacecraft, CubeSat, Mothership
from DSRC.simulation import (
    SimulationHistoryTimestep,
    SimulationHistory,
    SimulationHistoryMData,
)
from DSRC.simulation.communication import CommsSimManager, Message
from DSRC.simulation.samples import Sample
from dataclasses import dataclass
import numpy as np
import logging
import ray
from shortuuid import uuid
from typing import TypedDict
import itertools
from mpmath import mpf
from abc import ABC, abstractmethod
from time import time


class CubeSatConfig(TypedDict):
    """Configuration for simulated cubesats."""

    fuel_capacity: float


class MothershipConfig(TypedDict):
    """Configuration for simulated mothership."""

    initial_position: np.ndarray
    """The starting position of the craft."""
    cubesat_capacity: int
    """How many cubesats can be docked at once."""
    fuel_capacity: float
    """The total fuel available for the mothership."""


class SimulationConfig(TypedDict):
    """Configuration for a simulation."""

    mothership_config: list[MothershipConfig]
    """Mothership configuration.

    The length of the list determines how many
    motherships are in the simulation
    """
    cubesat_config: list[CubeSatConfig]
    """Cubesat configuration.

    If the list is 1-length and there are > 1 motherships
    in the simulation it's assumed that all motherships
    have the same loadout.

    Else then each mothership needs it's corresponding config.
    If this is not the case an exception will be thrown.
    """
    timestep: float
    """Delta in sim time."""


class _CraftMsgIterator:
    """Private module-level class for iteration over messages.

    Allows for iterating over (craft, msg) tuple pairs as
    this iterator will return all messages for all crafts
    in the order of crafts then in the order of messages in
    that craft's queue
    """

    def __init__(self, crafts: list[Spacecraft]):
        # print("="*50)
        self._crafts = [c for c in filter(lambda c: c.has_msg, crafts)]
        self._idx = 0
        # print("ID: Num msgs")
        # for c in self._crafts:
        # print(f"{c.id}: {c.msg_queue_size}")
        # print("")

    def __iter__(self):  # noqa D
        return self

    def __next__(self):  # noqa D
        if len(self._crafts) == 0:  # There's nothing to iterate over
            raise StopIteration
        c = self._crafts[self._idx]
        if not c.has_msg:  # No more msgs for this craft
            # print(f"No more msgs for {c.id}")
            self._idx += 1
            if self._idx == len(self._crafts):  # No more crafts left to iter over
                raise StopIteration
            c = self._crafts[self._idx]
            # print(f"{c.id} has a msg. It has {c.msg_queue_size} left")
        msg = c.get_msg()
        if msg is None:
            breakpoint()
        return (c, *msg)


class Simulation(ABC):
    """A Simulation of DSRC.

    This class is created as a Ray actor so that
    multiple simulations may be run in parallel.
    """

    _simlogger: logging.Logger
    """Class logger."""
    _id: str
    """UUID of this simulation."""
    _crafts: dict[str, Spacecraft] = dict()
    """Crafts which exist in the simulation.


    Specifically these are crafts which are deployed.
    CubeSats which are docked in the mothership
    will not show here. It's expected that a cubesat
    deployment does not retain state information from
    the previous deployment, i.e., the mothership
    instantiates a new CubeSat object which can be
    added to this method. If state is to be retained
    it's up to the mothership to handle that not the sim.
    """
    _samples: list[Sample] = []
    """The known samples which can be captured."""
    _cubesat_configs: dict[str, CubeSatConfig] = dict()
    """Mapping between a mothership's ID and the cubesat configs."""
    _history: list[SimulationHistoryTimestep] = []
    """List of sim timestep histories."""
    _metadata: SimulationHistoryMData
    """Metadata about this sim."""
    _simtime: mpf
    """The sim time."""
    _simdt: mpf
    """The simulation update period."""
    _comms_manager: CommsSimManager
    """Object to manage simulated communications."""

    def __init__(self, config: SimulationConfig, parentLogger: logging.Logger):
        """Initialize the simulation.

        Initialization occurs with a SimulationConfig dictionary.
        See that class doc string for member details.
        """
        now = time()
        self._id = uuid()
        self._simlogger = logging.getLogger(f"{parentLogger.name}.simulation.{self.id}")
        self._simlogger.debug(f"Creating simulation {self.id}")
        self._simdt = mpf(config["timestep"])
        self._comms_manager = CommsSimManager(self._simlogger)
        self._simtime = mpf(0.0)

        if len((cs_config := config["cubesat_config"])) == 1:
            self._simlogger.debug(
                "Got one cubesat config. Duplicating %s times",
                len(config["mothership_config"]),
            )
            cs_configs = [cs_config[0] for _ in config["mothership_config"]]
        else:
            if (nc := len(cs_config)) != (nm := len(config["mothership_config"])):
                raise ValueError(
                    f"Got {nc} cubesat configs and {nm} mothership configs."
                )
            cs_configs = cs_config

        # Instantiate all mothership(s)
        for ms_config, cs_config in zip(config["mothership_config"], cs_configs):
            ms = Mothership(
                ms_config["initial_position"],
                ms_config["cubesat_capacity"],
                self._simlogger,
                ms_config["fuel_capacity"],
            )
            self._crafts[ms.id] = ms
            self._simlogger.debug("Added mothership %s at %s", ms.id, ms.position)
            self._cubesat_configs[ms.id] = cs_config

        self._metadata = {
            "max_num_crafts": len(self.crafts),
            "max_num_samples": 0,
            "total_iters": 0,
            "craft_ids": set(self.crafts.keys()),
            "id": self.id,
            "time_start": now,
        }

        self._simlogger.debug(
            "Simulation sucesfully initialized with parameters %s", config
        )

    def run(self) -> SimulationHistory:
        """Run the simulation to completion.

        This is the main entrypoint to the simulation which
        starts and executes the main simulation loop. The loop
        runs until the termination condition is met (determined
        through the self._is_terminated() method) and, at each
        iteration, calls the self._update() method to update
        the simulation.

        Returns the simulation's history as a SimulationHistory
        TypedDict.
        """
        self._simlogger.info("Simulation loop starting")
        while not self._is_terminated():
            self._update()
        self._metadata["time_end"] = time()
        self._metadata["sim_time"] = self.simtime
        self._simlogger.info("Simulation loop ended at time %s", self.simtime)
        return {"history": self._history, "metadata": self._metadata}

    def _update(self) -> None:
        """Run one update loop.

        This function steps the simulation by one
        iteration and advances the sim time by dt.

        The update loop performs the following steps, in order:
        (1) Updates the kinematic state for each entity in the
            simulation. Entities are spacecraft and samples.
        (2) Call the self._planning_step() method which should
            perform some planning and/or coordination for
            the spacecraft in the simulation.
        (3) Call the self._update_samples method to possibly
            add more samples to the simulation
        (4) Update the simulation's metadata for later processing
        (5) Increase the simulation time by dt
        (6) Call the update loop for the SimCommsManager object
            which may deliver messages which have been in transmission.
            Any transmissions which are finalized will appear in the
            spacecraft's message queue on the next iteration.
        (7) Update the simulation history for later processing.
        """
        for entity in self.entities_iter:
            entity.update_kinematics(self.dt)
        self._planning_step()
        self._update_samples()

        if self._metadata["max_num_samples"] < len(self._samples):
            self._metadata["max_num_samples"] = len(self._samples)
        self._metadata["craft_ids"] |= set(self.crafts.keys())  # Update craft IDs
        if (nc := len(self.crafts)) > self._metadata["max_num_crafts"]:
            self._metadata["max_num_crafts"] = nc  # Update max # of craft at one iter
        self._simtime += self.dt
        # Update the comms simulation
        self._crafts = self._comms_manager.update(self.simtime, self.dt, self.crafts)
        self._history.append(
            {
                "time": self.simtime,
                "craft_positions": {c.id: c.position for c in self.crafts.values()},
                "craft_types": {
                    c.id: "CubeSat" if type(c) is CubeSat else "Mothership"
                    for c in self.crafts.values()
                },
                "sample_positions": [s.position.copy() for s in self.samples],
            }
        )
        self._metadata["total_iters"] += 1

    @abstractmethod
    def _is_terminated(self) -> bool:
        """Determine if a simulation is terminated or not.

        This is an abstract method whcih must be overriden in
        child classes.

        This function is called in the main simulation loop
        to determine if the simulation is terminated.

        Return: True if terminated else False
        """
        pass

    @abstractmethod
    def _planning_step(self) -> None:
        """Planning step for all craft in the simulation."""
        pass

    @abstractmethod
    def _update_samples(self) -> None:
        """Potentially add more known samples to the simulation."""
        pass

    def _craft_msg_iterator(self, T):
        if T is CubeSat:
            crafts = self.cubesats.values()
        elif T is Mothership:
            crafts = self.motherships.values()
        elif T is Spacecraft:
            crafts = self.crafts.values()
        else:
            raise ValueError(f"Unknown type {T}")
        return _CraftMsgIterator(crafts)

    @property
    def id(self) -> str:  # noqa D
        return self._id

    @property
    def crafts(self) -> dict[str, Spacecraft]:
        """Get all crafts in the simulation by their ID."""
        return self._crafts

    @property
    def entities_iter(self):
        """Return a chained iterator through all entities.

        This includes the spacecrafts and the smaples.
        """
        return itertools.chain(self._crafts.values(), self._samples)

    @property
    def motherships(self) -> dict[str, Mothership]:
        """Get all the crafts which are motherships."""
        return self._get_crafts_by_type(Mothership)

    @property
    def num_motherships(self) -> int:  # noqa D
        return len(self.motherships)

    @property
    def mothership_msg_iterator(self):  # noqa D
        return self._craft_msg_iterator(Mothership)

    @property
    def cubesats(self) -> dict[str, CubeSat]:
        """Get all the crafts which are cubesats."""
        return self._get_crafts_by_type(CubeSat)

    @property
    def num_cubesats(self) -> int:  # noqa D
        return len(self.cubesats)

    @property
    def cubesat_msg_iterator(self):  # noqa D
        return self._craft_msg_iterator(CubeSat)

    @property
    def craft_msg_iterator(self):  # noqa D
        return self._craft_msg_iterator(Spacecraft)

    @property
    def cubesat_configs(self) -> dict[str, CubeSatConfig]:  # noqa D
        return self._cubesat_configs

    @property
    def samples(self) -> list[Sample]:  # noqa D
        return self._samples

    @property
    def num_samples(self) -> int:  # noqa D:
        return len(self.samples)

    @property
    def simtime(self) -> float:  # noqa D
        return self._simtime

    @property
    def dt(self) -> float:  # noqa D
        return self._simdt

    @property
    def iter(self) -> int:  # noqa D
        return self._metadata["total_iters"]

    def _get_crafts_by_type(self, T):  # noqa D
        return {id: c for id, c in self.crafts.items() if type(c) is T}
