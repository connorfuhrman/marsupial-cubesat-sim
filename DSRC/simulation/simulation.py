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


class Simulation:
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


@ray.remote
class SimulationActor(Simulation):
    """A Simulation class as a Ray Actor.

    This simply inherits from Simulation and
    makes no changes. This allows Simulations
    to be run in a Ray session.
    """

    pass


def _make_circle(center: np.ndarray, rad: float, npoints: int = 100) -> np.ndarray:
    """Make a circle centered around some point with a given radius.

    The circle lies in a horizontal plane at z = center[2].
    """
    theta = np.linspace(0, 2 * np.pi, npoints)
    xpnts = rad * np.cos(theta)
    ypnts = rad * np.sin(theta)

    arr = np.empty((npoints, 3), dtype=float)
    for i, (x, y) in enumerate(zip(xpnts, ypnts)):
        arr[i, :] = np.array([x, y, center[2]])
    return arr


def _del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del l[i]


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _find_repeats(arr: np.ndarray) -> np.ndarray:
    """Find indices of repeat values in an array.

    Args:
        arr (np.ndarray): An array to find repeat values in.

    Returns:
        np.ndarray: An array of indices into arr which are the values which
            repeat.

    From https://stackoverflow.com/a/67780952/10302537
    """

    arr_diff = np.diff(arr, append=[arr[-1] + 1])
    res_mask = arr_diff == 0
    arr_diff_zero_right = np.nonzero(res_mask)[0] + 1
    res_mask[arr_diff_zero_right] = True
    return np.nonzero(res_mask)[0]


def _test():  # noqa C901: I know the cyclomatic complexity is too large. This is just an intermediate test
    import argparse as ap
    from DSRC.simulation import animate_simulation
    from DSRC.simulation.communication import messages as msgs
    from DSRC.simulation.utils import save_json_file

    parser = ap.ArgumentParser()
    parser.add_argument(
        "--num_workers",
        type=int,
        help="How many Ray workers. If 1 run without Ray.",
        default=1,
    )
    parser.add_argument(
        "--cubesat_radius", type=float, help="The radius path to fly", default=12.5
    )
    parser.add_argument(
        "--num_cubesats",
        type=int,
        help="The number of cubesats (only applicable to non-Ray runs",
        default=1,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="The max number of samples in a simulation.",
        default=15,
    )
    parser.add_argument(
        "--mp4_file", type=str, help="MP4 file to save the animation.", default=None
    )
    parser.add_argument(
        "--no_animation", help="Don't do an animation just run", action="store_true"
    )
    parser.add_argument(
        "--sim_history_save",
        help="JSON file to save the sim history",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    class NoMoreSamples(Exception):
        pass

    class Updater:
        """An example of a centralized planner."""

        def __init__(self):
            self.iters = 0
            self.logger = None
            self.num_cubesats_deployed = 0
            self.cubesat_sample_assignments = dict()

        def do_cubesat_deploy(self, mothership, crafts, config):
            offset = np.array([0, 0, 5 * (self.num_cubesats_deployed + 1)], dtype=float)
            deploy_pos = mothership.position - offset
            cs = CubeSat(deploy_pos, config["fuel_capacity"], 0.95, self.logger)
            # waypoints = _make_circle(deploy_pos, args.cubesat_radius, npoints=25)
            # for w in waypoints:
            #     cs.add_waypoint(w) # + np.random.normal(0, 0.25, size=3))
            # cs.add_waypoint(mothership.position)
            crafts[cs.id] = cs
            mothership.deploy_cubesat()
            self.num_cubesats_deployed += 1
            return crafts

        def assign_sample_random(self, cubesats, mothership, samples):
            # Get idxs of input that are not assigned
            idxs = [
                i
                for i, s in enumerate(samples)
                if s not in self.cubesat_sample_assignments.values()
            ]
            # Loop over all cubesats without an assignment
            for c in filter(
                lambda c: c.id not in self.cubesat_sample_assignments, cubesats
            ):
                idx = np.random.choice(idxs, replace=False)
                self.cubesat_sample_assignments[c.id] = samples[idx]
                c.add_waypoint(samples[idx].position.copy())

            return cubesats, samples

        def mothership_assign_sample_min_dist(self, cubesats, mothership, samples):
            waypoints_to_assign = dict()
            for c in filter(lambda cs: cs.curr_waypoint is None, cubesats):
                dists = {
                    np.linalg.norm(c.position - s.position): i
                    for i, s in enumerate(samples)
                }
                while True:
                    dist_vals = list(dists.keys())
                    key = dist_vals[np.argmin(dist_vals)]
                    idx = dists[key]
                    sample = samples[idx]
                    if sample not in self.cubesat_sample_assignments.values():
                        waypoints_to_assign[c] = sample
                        break
                    del dists[key]  # The sample was assigned so onto the next-closest
                    if len(dists) == 0:
                        break

            # for c, sample in waypoints_to_assign.items():
            # self.cubesat_sample_assignments[c.id] = sample
            # c.add_waypoint(sample.position)

            def make_msg(c, sample):
                self.cubesat_sample_assignments[c.id] = sample
                return msgs.SampleAquireCommand(
                    tx_id=mothership.id,
                    rx_id=c.id,
                    timestamp=0.0,
                    sample_pos=sample.position.copy(),
                )

            msgs_to_send = [
                make_msg(c, sample) for c, sample in waypoints_to_assign.items()
            ]

            return cubesats, mothership, msgs_to_send, samples

        def do_try_capture(self, cubesats, samples):
            cs_id_map = {c.id: c for c in cubesats}
            to_delete = []
            for cubesat_id, samp in self.cubesat_sample_assignments.items():
                c = cs_id_map[cubesat_id]
                if np.linalg.norm(c.position - samp.position) <= 1:
                    # Attempt a capture if close enough
                    if c.attempt_sample_capture(samp):
                        c.drop_curr_waypoint()
                        to_delete.append(c.id)
                        samples = [s for s in samples if s is not samp]
                        if (
                            len(samples) == 0
                        ):  # If there's no more samples wait to recall this func to replan
                            break
            for id in to_delete:
                del self.cubesat_sample_assignments[id]
            return cubesats, samples

        def do_all_cubsat_msg_callbacks(self, cubesats):
            cs_id_map = {c.id: c for c in cubesats}
            for csat in filter(lambda c: c.msg_queue_size > 0, cubesats):
                while (timestamped_msg := csat.get_msg()) is not None:
                    msg = timestamped_msg[1]
                    if msgs.Message.is_type(msg, msgs.SampleAquireCommand):
                        id = msg.rx_id
                        waypoint = msg.msg["sample_pos"]
                        cs_id_map[id].add_waypoint(waypoint, front=True)
                    else:
                        raise RuntimeError("Unrecognized message type")
            return cubesats

        def __call__(self, crafts, samples, cs_configs, logger):
            cubesats = [c for c in crafts.values() if type(c) is CubeSat]
            mothership = [c for c in crafts.values() if type(c) is Mothership][0]
            cubesats = self.do_all_cubsat_msg_callbacks(cubesats)
            msgs = None
            if self.iters == 0:
                self.logger = logger
                # Deploy a cubesat
                while mothership.can_deploy_cubesat:
                    crafts = self.do_cubesat_deploy(
                        mothership, crafts, cs_configs[mothership.id]
                    )
            elif len(samples) > 0:
                # Assign each craft to capture the closest sample
                # cubesats, samples = self.assign_sample_random(cubesats, mothership, samples)

                # As the mothership, command each craft to a specific asteroid
                (
                    cubesats,
                    mothership,
                    msgs,
                    samples,
                ) = self.mothership_assign_sample_min_dist(
                    cubesats, mothership, samples
                )

                # Attempt any sample captures we can
                cubesats, samples = self.do_try_capture(cubesats, samples)
            elif self.iters > 100:
                for c in cubesats:
                    c.add_waypoint(
                        mothership.position, front=True
                    )  # Go dock if there's no samples
                    # Can this cubesat dock?
                    if np.linalg.norm(c.position - mothership.position) < 0.5:
                        mothership.dock_cubesat(c)
                        del crafts[c.id]  # Remove from the sim's list of crafts

            self.iters += 1
            return crafts, msgs, samples

    updater = Updater()

    def ejector(time, samples, Plogger):
        if np.random.uniform() > 0.75 and len(samples) < args.max_samples and time < 60:
            logger = logging.getLogger(f"{Plogger.name}.ParticleEjector")
            nsamps = np.random.randint(1, 15)
            logger.debug("Adding %s samples to the simulation at %s", nsamps, time)
            return [make_sample() for _ in range(nsamps)]
        # if time == 0.0:
        #     print("Making samples")
        #     return [make_sample() for _ in range(2)]
        else:
            return []

    def make_sample():
        return Sample(
            weight=np.random.uniform(5, 10),
            value=np.random.uniform(1, 10),
            position=np.random.uniform(-15, 15, 3),
            velocity=np.zeros(3),
        )

    max_sim_time_min = 180

    def terminator(time, crafts, samples):
        # cs = [c for c in crafts if type(c) is CubeSat]
        # return len(cs) == 0 or time >= 60*max_sim_time_min
        # return (len(samples) == 0 and time > 1) or \
        #     time > max_sim_time_min * 60
        if len(crafts) == 1:
            print("All cubesats docked!! Simulation ending.")
            return True
        elif time > max_sim_time_min * 60:
            print("Timeout!")
            return True
        else:
            return False

    config: SimulationConfig = {
        "mothership_config": [
            {
                "initial_position": np.array([0, 0, 0], dtype=float),
                "cubesat_capacity": args.num_cubesats,
                "fuel_capacity": 100,
            },
        ],
        "cubesat_config": [
            {"fuel_capacity": 10},
        ],
        "planner": updater,
        "particle_ejector": ejector,
        "terminator": terminator,
        "timestep": 0.05,
    }

    if args.num_workers > 1:

        def make_actor():
            _config = config.copy()
            _config["mothership_config"][0]["cubesat_capacity"] = np.random.randint(
                5, 20
            )
            return SimulationActor.remote(_config)

        ray.init()
        ncpus = int(ray.available_resources()["CPU"])
        if ncpus == 1:
            raise RuntimeError(
                "You're trying to use Ray on a single-core machine! "
                "Set --num_workers=1 next time"
            )
        sims_history = []
        # Manage remote workers
        actors = [make_actor() for _ in range(ncpus)]
        refs_to_actors = {a.run.remote(): a for a in actors}
        while len(sims_history) != args.num_workers:
            ready, refs = ray.wait(list(refs_to_actors.keys()))
            for ref in ready:
                sims_history.append(ray.get(ref))
                del refs_to_actors[ref]
            if len(sims_history) + len(refs) < args.num_workers:
                actor = make_actor()
                refs_to_actors[actor.run.remote()] = actor

        if not args.no_animation:
            print(f"Animating {args.num_workers} simulations")
            animate_simulation(sims_history, args.mp4_file)
        if args.sim_history_save is not None:
            to_save = {s["metadata"]["id"]: s for s in sims_history}
            save_json_file(to_save, args.sim_history_save)
    else:
        try:
            sim = Simulation(config)
            sim_history = sim.run()
            if not args.no_animation:
                print("Animating ...")
                animate_simulation([sim_history], args.mp4_file)
        except Exception:
            import pdb, traceback, sys  # noqa E401

            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)


if __name__ == "__main__":
    _test()
