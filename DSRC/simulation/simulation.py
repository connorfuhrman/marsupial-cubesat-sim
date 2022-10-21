"""Simulation of Decentralized Sample Recovery Coordination."""


from DSRC.simulation.spacecraft import Spacecraft, CubeSat, Mothership
from DSRC.simulation import SimulationHistoryTimestep, SimulationHistory, SimulationHistoryMData
from dataclasses import dataclass
import numpy as np
import logging
import ray
from shortuuid import uuid
from typing import TypedDict


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
    planner: callable
    """Callable which accepts the crafts as an argument and returns them back updated."""
    terminator: callable
    """Callable which determins if the simulation is over."""



@dataclass
class Sample:
    """Standin as a sample class for now."""

    weight: float = 10.0
    """Sample weight in g."""
    value: float = 5.0
    """Value in [0, 10] rating how valuable the sample is."""
    position: np.ndarray = None
    """The sample's location in 3-space."""


class Simulation:
    """A Simulation of DSRC.

    This class is created as a Ray actor so that
    multiple simulations may be run in parallel.
    """

    _logger: logging.Logger
    """Class logger."""
    _id: str
    """UUID of this simulation."""
    _crafts: set[Spacecraft] = set()
    """Crafts which exist in the simulation.


    Specifically these are crafts which are deployed.
    CubeSats which are docked in the mothership
    will not show here. It's expected that a cubesat
    deployment does not retain state information from
    the previous deployment, i.e., the mothership
    instantiates a new CubeSat object which can be
    added to this method.
    """
    _samples = None
    """The known samples which can be captured."""
    _cubesat_configs: dict[str, CubeSatConfig] = dict()
    """Mapping between a mothership's ID and the cubesat configs."""
    _history: list[SimulationHistoryTimestep] = []
    """List of sim timestep histories."""
    _metadata: SimulationHistoryMData
    """Metadata about this sim."""
    _simtime: float = 0.0
    """The sim time."""
    _simdt: float
    """The simulation update period."""
    _planner: callable
    """Callable which performs planning for each craft."""
    _termiantor: callable
    """Callable which determines if the simulation is over."""

    def __init__(self, config: SimulationConfig):
        """Initialize the simulation.

        Initialization occurs with a SimulationConfig dictionary.
        See that class doc string for member details.
        """
        logging.debug("Creating simulation with config %s", config)
        self._id = uuid()
        logger = logging.getLogger(f"root.simulation.{self.id}")
        self._logger = logger
        self._logger.info(f"Creating simulation {self.id}")
        self._logger.setLevel(logging.WARNING)
        self._planner = config['planner']
        self._terminator = config['terminator']
        self._simdt = config['timestep']

        if len((cs_config := config['cubesat_config'])) == 1:
            cs_configs = [cs_config[0] for _ in config['mothership_config']]
        else:
            if (nc := len(cs_config)) != (nm := len(config['mothership_config'])):
                raise ValueError(f"Got {nc} cubesat configs and {nm} mothership configs.")
            cs_configs = cs_config

        # Instantiate all mothership(s)
        for ms_config, cs_config in zip(config['mothership_config'], cs_configs):
            ms = Mothership(ms_config['initial_position'],
                            ms_config['cubesat_capacity'],
                            self._logger,
                            ms_config['fuel_capacity'])
            self._crafts.add(ms)
            self._logger.debug("Added mothership %s as %s",
                               ms.id, ms.position)
            self._cubesat_configs[ms.id] = cs_config

        self._metadata = {
            'max_num_crafts': len(self.crafts),
            'total_iters': 0,
            'craft_ids': {c.id for c in self.crafts},
            'id': self.id,
        }

    def run(self) -> SimulationHistory:
        """Run the simulation."""
        self._logger.info("Simulation beginning")
        while True:
            self._update()
            if self._terminator(self.simtime, self.crafts, None):
                break
        self._logger.info("Simulation ended at time %s", self.simtime)
        return {"history": self._history, "metadata": self._metadata}

    def _update(self) -> None:
        """Run one update loop.

        This function steps the simulation by one
        iteration and advances the sim time by dt.
        The kinematics are updated from the previous planning
        step and then another planning session is conducted.

        Planning is done by calling the planner which
        is a callable passed to this function
        which takes the crafts in the simulation as
        an argument. This function must return
        the set of crafts back with whatever changes are
        made to the crafts state in that return variable.
        """
        for c in self.crafts:
            c.update_kinematics(self.dt)
        self._crafts = self._planner(self._crafts, None, self.cubesat_configs, self._logger)
        self._metadata['craft_ids'] |= {c.id for c in self.crafts}
        if (nc := len(self.crafts)) > self._metadata['max_num_crafts']:
            self._metadata['max_num_crafts'] = nc
        self._simtime += self.dt
        self._history.append(
            {
                "time": self.simtime,
                "craft_positions": {c.id: c.position for c in self.crafts},
                "craft_types": {c.id: "CubeSat" if type(c) is CubeSat else "Mothership"
                                for c in self.crafts}
            })
        self._metadata['total_iters'] += 1

    @property
    def id(self) -> str:  # noqa D
        return self._id

    @property
    def crafts(self) -> set[Spacecraft]:
        """Get all crafts in the simulation."""
        return self._crafts

    @property
    def motherships(self) -> set[Mothership]:
        """Get all the crafts which are motherships."""
        return self._get_crafts_by_type(Mothership)

    @property
    def cubesats(self) -> set[CubeSat]:
        """Get all the crafts which are cubesats."""
        return self._get_crafts_by_type(CubeSat)

    @property
    def cubesat_configs(self) -> dict[str, CubeSatConfig]:  # noqa D
        return self._cubesat_configs

    @property
    def simtime(self) -> float:  # noqa D
        return self._simtime

    @property
    def dt(self) -> float:  # noqa D
        return self._simdt

    @property
    def iter(self) -> int:  # noqa D
        return self._metadata['total_iters']

    def _get_crafts_by_type(self, T):  # noqa D
        return {c for c in self.crafts if type(c) is T}


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
    theta = np.linspace(0, 2*np.pi, npoints)
    xpnts = rad * np.cos(theta)
    ypnts = rad * np.sin(theta)

    arr = np.empty((npoints, 3), dtype=float)
    for i, (x, y) in enumerate(zip(xpnts, ypnts)):
        arr[i, :] = np.array([x, y, center[2]])
    return arr


def _test():
    import argparse as ap
    from DSRC.simulation import animate_simulation

    parser = ap.ArgumentParser()
    parser.add_argument("--num_workers",
                        type=int,
                        help="How many Ray workers. If 1 run without Ray.",
                        default=1)

    args = parser.parse_args()

    class Updater:
        def __init__(self):
            self.first_update = True
            self.logger = None
            self.num_cubesats_deployed = 0

        def do_cubesat_deploy(self, mothership, crafts, config):
            offset = np.array([0, 0, 2 * (self.num_cubesats_deployed + 1)], dtype=float)
            deploy_pos = mothership.position - offset
            cs = CubeSat(deploy_pos, config['fuel_capacity'],
                         0.95, self.logger)
            waypoints = _make_circle(deploy_pos, 7)
            for w in waypoints:
                cs.add_waypoint(w + np.random.normal(0, 0.5, size=3))
            cs.add_waypoint(mothership.position)
            crafts.add(cs)
            mothership.deploy_cubesat()
            self.num_cubesats_deployed += 1
            return crafts

        def __call__(self, crafts, samples, cs_configs, logger):
            cubesats = [c for c in crafts if type(c) is CubeSat]
            mothership = [c for c in crafts if type(c) is Mothership][0]
            for c in cubesats:
                if np.linalg.norm(c.position - mothership.position) < 1:
                    mothership.dock_cubesat(c)
                    crafts.remove(c)
            if self.first_update:
                self.logger = logger
                # Deploy a cubesat
                while mothership.can_deploy_cubesat:
                    crafts = self.do_cubesat_deploy(mothership, crafts, cs_configs[mothership.id])
                self.first_update = False
            return crafts

    updater = Updater()

    def terminator(time, crafts, samples):
        cs = [c for c in crafts if type(c) is CubeSat]
        return len(cs) == 0 or time > 10000

    config: SimulationConfig = {
        'mothership_config': [
            {
                'initial_position': np.array([0, 0, 0], dtype=float),
                'cubesat_capacity': 1,
                'fuel_capacity': 100
            },
        ],
        'cubesat_config': [
            {
                'fuel_capacity': 10
            },
        ],
        'planner': updater,
        'terminator': terminator,
        'timestep': 0.1,
    }

    if args.num_workers > 1:
        def make_actor(n):
            _config = config.copy()
            _config['mothership_config'][0]['cubesat_capacity'] = np.random.randint(2, 10)
            return SimulationActor.remote(config)
        sims = [make_actor(n) for n in range(args.num_workers)]
        sims_history = ray.get([s.run.remote() for s in sims])
        print(f"Animating {args.num_workers} simulations")
        animate_simulation(sims_history)
    else:
        sim = Simulation(config)
        sim_history = sim.run()
        animate_simulation([sim_history])


if __name__ == '__main__':
    _test()
