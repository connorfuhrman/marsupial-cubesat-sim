"""Experiment to capture a particle ejection from the asteroid Bennu.

Samples are pregenerated using the MATLAB script authored by Dr. Leonard Vance.
Their trajectories are read into the simulation when needed. 

The experiment begins with some number of cubesats located at the simulated particle
ejections as generated by the MATLAB simulation. The locations are determined by the
first timestep where all samples are greater or equal to some straight-line distance
from the origin of the asteroid.
"""

from DSRC.simulation import (
    Simulation,
    SimulationConfig
)

from DSRC.simulation.spacecraft import Mothership, CubeSat
from DSRC.simulation.communication import messages as msgs
from DSRC.simulation.samples import Sample
#from DSRC.experiments.models.bennu_particle_ejection_return import Model

import logging
import numpy as np
import pathlib as pl
import pickle
import random
from typing import TypedDict
from enum import Enum, auto
from pprint import pformat
import random


class Config(TypedDict):
    """Configuration for the Bennu Particle Return experiment."""

    simulation_config: SimulationConfig
    """Basic simulation config."""

    bennu_pos: np.ndarray
    """Position of Bennu in the simulation."""

    transmission_freq: float
    """Rate in Hz the craft will transmit update messages."""

    particle_database: pl.Path
    """Path to pickled database of particle ejections."""

    action_space_rate: float
    """Minimum update rate for actions."""

    action_space_dock_dist: float
    """Minimum distance from mothership to dock."""

    action_space_waypoint_dist: float
    """Magnitude of distance for setting waypoint."""

    model: "Model"
    """Model used as Q function approximator."""

    num_iters_calc_reward: int
    # Number of iterations to pass before calculating a reward

    min_num_samples: int
    # The minimum number of samples (and therefore cubesats)

    max_num_samples: int
    # The maximum number of samples (and therefore cubesats)


class OutOfData(Exception):
    """Exception to trigger that there's no more particle ejection data."""
    pass


class ParticleDatabase:
    """Helper class to manage particle data."""

    def __init__(self, file: pl.Path, bennu_orig: np.ndarray,
                 logger: logging.Logger):
        """Initialize with pickle file to raw data and logger."""
        pass
        # self.logger = logging.getLogger(logger.name + ".database")
        # self.logger.info("Loading pickled data from %s", file)
        # self.data = pickle.load(file)
        # if (t := type(self.data)) is not dict:
        #     raise ValueError(f"Expected data as dict but got {t}")
        # self.logger.info("Got %s datapoints", len(self.data))
        self.bennu_orig = bennu_orig

    def draw(self, num_particles: int, min_dist: float):
        """Draw some number of particles that are all min_dist away.

        The samples are initialized with a random value between 0 and 10.
        """
        # samples = []
        # while len(samples != num_particles):
        #     traj = np.array(self.random())
        pos_from_bennu = lambda: np.random.uniform(-5.0, 5.0, size=(3,))
        samples = [Sample(0.0, np.random.uniform(0.0, 10.0),
                          (pos_from_bennu() + self.bennu_orig).copy(),
                          np.zeros(3))
                   for _ in range(num_particles)]
        return samples


    def random(self, replace: bool = False):
        """Randomly draw from the dataset.

        This function permutes the dataaset. If not replacing then
        the data is deleted from the set.
        """
        if len(self.items) == 0:
            raise OutOfData
        k, v = random.choice(list(self.data.items()))
        if not replace:
            del self.data[k]
        return v


class StatusBeacon:
    """Transmit status at a regular interval.

    This class is to assist regular transmissions of craft status.
    """

    def __init__(self, rate: float):  # noqa D
        self.sec_btwn = 1.0/rate
        self.last_tx_time = 0.0

    def update(self, craft: CubeSat, sim_time: float):
        """Return a status message at the appropriate rate."""
        if (sim_time - self.last_tx_time) >= self.sec_btwn:
            self.last_tx_time = sim_time
            return craft.get_state_msg(sim_time)
        return None


class ObservationSpace:
    """A cubesat agent's observation space."""

    def __init__(self, mship: Mothership, craft: CubeSat):  # noqa D
        self.state = dict()
        self.mothership_pos = mship.position  # mothership is stationary
        self.logger = logging.getLogger(craft.logger.name + ".observation_space")
        self.cur_pos = craft.position
        self.mothership_relative = self.relative_spherical(self.mothership_pos)

    def update(self, m, craft: CubeSat):
        """Update the observation space.

        Updates the state knowledge given any new messages
        from the CubeSat.
        """
        self.cur_pos = craft.position
        self.mothership_relative = self.relative_spherical(self.mothership_pos)
        # TODO should I be cleaning up the states for craft that we've not heard from
        # in some period of time?
        if msgs.Message.is_type(m, msgs.CubeSatState):
            tx_id = m.msg['tx_id']
            m.msg['position'] = self.relative_spherical(m.msg['position'])
            self.state[tx_id] = m.msg.copy()
            self.logger.debug("Updating state for craft %s. "
                              "Received state is: \n%s", tx_id, pformat(m.msg))
        elif msgs.Message.is_type(m, msgs.CubeSatDocked):
            if (id := m.msg['id']) in self.state:
                self.logger.info("Craft %s is known to be docked.", id)
                del self.state[id]
        else:
            raise ValueError("Unknown message type")

    def relative_spherical(self, other_position: np.ndarray) -> np.ndarray:
        """Convert other craft's position.

        The other craft's position is reported as relative spherical coordiantes
        from this craft's current position.
        """
        relative = other_position - self.cur_pos
        x, y, z = relative

        r = np.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2))
        theta = np.arctan(y/x)
        phi = np.arccos(z/r)

        return np.array([r, theta, phi])
        

    def observations(self, time: float):
        """Return set of observations at the current time."""
        # The 'position' key is changed to be relative to this cubesat's
        # current position
        obs = [(s['fuel_level'], s['sample_value'], time - s['timestamp'], *s['position'])
               for s in self.state.values()]
        random.shuffle(obs)
        return obs


class ActionSpace:
    """An agents action space.

    The action is determined by querying the neural network model
    for the Q value.
    """

    vals = Enum('vals', [
        'noop',  # Do nothing
        'dock',  # Put a waypoint at the mothership
        'wp_u',  # Waypoint directly above
        'wp_d',  # Waypoint directly below
        # Waypoints to front-left
        'wp_fl_u',  # waypoint front-left +z
        'wp_fl_c',  # waypoint front-left
        'wp_fl_d',  # waypoint front-left -z
        # Waypoints to front
        'wp_f_u',
        'wp_f_c',
        'wp_f_d',
        # Waypoints to front-right
        'wp_fr_u',
        'wp_fr_c',
        'wp_fr_d',
        # Waypoints to right
        'wp_r_u',
        'wp_r_c',
        'wp_r_d',
        # Waypoints to back-right
        'wp_br_u',
        'wp_br_c',
        'wp_br_d',
        # Waypoints to back
        'wp_b_u',
        'wp_b_c',
        'wp_b_d',
        # Waypoints to back-left
        'wp_bl_u',
        'wp_bl_c',
        'wp_bl_d'
    ], start=0)

    def __init__(self, rate: float, min_dock_range: float, waypoint_dist: float,
                 craft_logger: logging.Logger):
        """Initialize action space.

        The waypoint_dist paramter determines the distance from the
        craft a waypoint will be set.

        The min_dock_range is the minimum distance between the craft
        and the mothership for a docking waypoint to be given. This
        action value only puts a waypoint at the mothership's location
        and it's assumed that once within range the craft will attempt
        to dock with the mothership.

        The rate is the minimum update time. If a craft is tracking a
        waypoint it must reach that waypoint before updating. Similarly,
        if the craft intends to dock it must finish that action. If the
        craft performed a noop another action will be determined only
        after 1.0/rate seconds has passed.
        """
        self.cur_action = None
        self.update_period = 1.0/rate
        self.next_update_time = 0.0
        self.min_dock_range = min_dock_range
        self.logger = logging.getLogger(craft_logger.name + ".action_space")

        self.above = np.array([0, 0, waypoint_dist], dtype=float)
        self.below = -1.0 * self.above
        self.right = np.array([0, waypoint_dist, 0], dtype=float)
        self.left = -1.0 * self.right
        self.forward = np.array([waypoint_dist, 0, 0], dtype=float)
        self.behind = -1.0 * self.forward

    def update(self, craft: CubeSat, mship: Mothership, sim_time: float, model, obs: ObservationSpace):
        """Get the next action given the model and observations."""

        def do_action():
            self.cur_action = self._q_value(model, obs, craft, sim_time)
            self._do_action(craft, mship)

        if self.cur_action is None:
            do_action()
        elif self.cur_action == ActionSpace.vals.noop and (sim_time >= self.next_update_time):
            self.next_update_time = sim_time + self.update_period
            do_action()
        elif craft.num_waypoints == 0:
            do_action()

    def _q_value(self, model, obs_space: ObservationSpace, craft: CubeSat, sim_time: float):
        if model is None:
            raise RuntimeError("No model!")
            val = np.random.randint(low=0, high=len(ActionSpace.vals))
            return ActionSpace.vals(val)
        my_state_msg = craft.get_state_msg(sim_time)
        # my_state = (my_state_msg['fuel_level'], my_state_msg['sample_value'],
        #             0.0, *my_state_msg['position'])
        obs = obs_space.observations(sim_time)
        # obs.append(my_state)
        if len(obs) == 0:
            # print(f"WARNING: Craft {craft.id} has no observations at {sim_time}")
            if sim_time > 60.0:
                raise RuntimeError("Did not get any obs after 1 minute")
            val = np.random.randint(low=0, high=len(ActionSpace.vals))
            return ActionSpace.vals(val)
        return model.get_action(obs, obs_space.mothership_relative)

    def _do_action(self, craft: CubeSat, mship: Mothership):
        """Perform the action in the action-space."""
        assert craft.num_waypoints == 0
        # Noop does nothing
        if self.cur_action == ActionSpace.vals.noop:
            return
        # If we can dock then set a waypoint at mothership else convert to a noop
        if self.cur_action == ActionSpace.vals.dock:
            if (dist := np.linalg.norm(craft.position - mship.position)) <= self.min_dock_range:
                self.logger.info("Craft %s is %sm away from mothership and is moving to dock",
                                 craft.id, dist)
                craft.add_waypoint(mship.position)
                return
            else:
                self.logger.debug("Craft %s wanted to dock but was %sm away.",
                                  craft.id, dist)
                self.cur_action = None
                raise BennuParticleReturn.InvalidDockingCmd()
        # If not a noop or a docking then it's a waypoint
        self.logger.debug("Setting waypoint for action %s", self.cur_action.name)
        self._do_waypoint(craft)

    def _do_waypoint(self, craft: CubeSat):
        # The most general case: set a waypoint at some offset depending on
        # the action space's value. The offset is applied based on the naming
        # scheme
        waypnt = craft.position
        assert 'wp' in self.cur_action.name
        if 'u' in self.cur_action.name:
            waypnt += self.above
        if 'd' in self.cur_action.name:
            waypnt += self.below
        if 'f' in self.cur_action.name:
            waypnt += self.forward
        if 'b' in self.cur_action.name:
            waypnt += self.behind
        if 'r' in self.cur_action.name:
            waypnt += self.right
        if 'l' in self.cur_action.name:
            waypnt += self.left

        craft.add_waypoint(waypnt)

    @staticmethod
    def num_states():
        """Return the number of possible actions."""
        return len(ActionSpace.vals)


class BennuParticleReturn(Simulation):
    """Bennu Particle Ejection Return Expriment.

    The experiment simulates the return of some number of cubestats after
    the cubesats captured ejected particles from the aseroid Bennu.
    """

    class CollisionEvent(Exception):
        """Exception event to end the experiment if we get a collision."""

        pass

    class InvalidDockingCmd(Exception):
        """Exception event when the action space commands an invalid dock."""

        pass

    def __init__(self, config: Config, controller_cls = ActionSpace, logger: logging.Logger = None):  # noqa D
        if logger is None:
            # Assume this is made within a Ray actor
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
        self.logger = logger

        self.config = config

        super().__init__(config['simulation_config'], self.logger)

        self.logger.info("Reading from database at %s", config['particle_database'])
        self.particle_data = ParticleDatabase(config['particle_database'],
                                              config['bennu_pos'],
                                              self.logger)

        self._initialize()
        self.initial_cubesat_ids = list(self.cubesats.keys())
        self.status_beacons = {c_id: StatusBeacon(self.config['transmission_freq'])
                               for c_id in self.cubesats.keys()}
        self.obs_spaces = {c_id: ObservationSpace(self.single_mothership, c)
                           for c_id, c in self.cubesats.items()}

        def action_space(craft):
            return controller_cls(self.config['action_space_rate'],
                                  self.config['action_space_dock_dist'],
                                  self.config['action_space_waypoint_dist'],
                                  craft.logger)

        self.action_spaces = {c_id: action_space(c)
                              for c_id, c in self.cubesats.items()}

        self.init_num_cubesats = self.num_cubesats
        self.max_sample_value = sum([c.sample_value for c in self.cubesats.values()])
        if self.max_sample_value == 0.0:
            self.logger.error("There were no captures. This is unusual!")
            self.max_sample_value = 1  # Set to 1 to guard against div by 0. If none we captured none will be returned
        self.num_cubesats_recovered = 0
        self.sample_value_recovered = 0.0
        self.distances_to_mothership = self._calc_dists_to_mothership()
        self.episode_reward = 0.0
        self.num_rewards_calculated = 0
        self.num_invalid_docking_commands = 0
        self.total_num_actions = 0

        self.model = self.config['model']
        if self.model == "expert_system":
            self.logger.info("Running with expert system. No model provided")
        if self.model is None:
            self.logger.warning("Got no model. Will do random actions")

    ################################################
    # Override methods required in base simulation #
    ################################################

    def _planning_step(self):
        self._update_craft_msgs()
        self._do_craft_action()
        if self.iter % self.config['num_iters_calc_reward'] == 0:
            self._calc_reward()
        mothership = self.single_mothership
        for c in self.cubesats.values():
            assert c.id in self.initial_cubesat_ids
            if np.linalg.norm(c.position - mothership.position) <= 0.5:
                mothership.dock_cubesat(c)
                self.num_cubesats_recovered += 1
                self.sample_value_recovered += c.sample_value
                self._send_cs_docked_msg(c.id)
                del self._crafts[c.id]

    def _is_terminated(self):
        """Determine termination event.

        Simulation is terminated when there are no more cubesats.
        """
        self._check_collision_event()
        return self.num_cubesats == 0

    def _update_samples(self):
        """Update all samples (no-op).

        This function does nothing since all samples are captured
        in the cubesats for this scenario.
        """
        pass

    ################################################
    # Helper Methods
    ################################################

    def _initialize(self):
        #n_samps = int(np.random.uniform(low=3, high=10))
        low = self.config['min_num_samples']
        high = self.config['max_num_samples']
        if low != high:
            n_samps = np.random.randint(low=low, high=high)
        else:
            n_samps = low
                                    
        self.logger.info("Initializing with %s samples", n_samps)
        samples = self.particle_data.draw(n_samps, 0.0)
        config = self._cubesat_configs[self.single_mothership.id]
        for s in samples:
            # Start cubesat at the sample's position. The cubesat
            # captured the sample with some given probability
            cs = CubeSat(s.position,
                         config['fuel_capacity'],
                         config['sample_capture_prob'],
                         self.logger)
            cs.attempt_sample_capture(s)
            # cs.add_waypoint(self.single_mothership.position)
            cs.vel_mag = 0.25
            self._crafts[cs.id] = cs
            self.logger.info("Cubesat %s is starting at %s "
                             "and has %s it's sample with value %s",
                             cs.id,
                             cs.position,
                             ("" if cs.has_sample else "not ") + "captured",
                             s.value)

    def _update_craft_msgs(self):

        def do_others(msg):
            others = filter(lambda id: id != msg['tx_id']
                            and id != self.single_mothership.id,
                            self.cubesats.keys())
            for o in others:
                m = msg.copy()
                m['rx_id'] = o
                self._comms_manager.send_msg(msgs.Message(m))

        for id, craft in self.cubesats.items():
            if (m := self.status_beacons[id].update(craft, self.simtime)) is not None:
                do_others(m)

        for cs, tx_time, msg in self.cubesat_msg_iterator:
            self.obs_spaces[cs.id].update(msg, cs)

    def _do_craft_action(self):
        """Query the action space for each agent.

        The action space's update function will only update the
        agent's actions when appropritae so it's safe to call at each iteration.
        """
        mothership = self.single_mothership
        self.num_invalid_docking_commands = 0
        for id, cs in self.cubesats.items():
            self.total_num_actions += 1
            try:
                self.action_spaces[id].update(cs, mothership, self.simtime, self.model, self.obs_spaces[id])
            except BennuParticleReturn.InvalidDockingCmd:
                self.num_invalid_docking_commands += 1

    def _check_collision_event(self):
        """Determine if a collision occured.

        Collisions are only assumed to occur near the mothership.
        A collision happens when two craft are within 5m of the
        mothership and are within 1m of each other.
        """
        mothership = self.single_mothership
        within_mship_range = [c for c in self.cubesats.values()
                              if np.linalg.norm(c.position - mothership.position) <= 2.5]

        def others(cs):
            return filter(lambda c: c.id != cs.id,
                          within_mship_range)

        for cs in within_mship_range:
            for o in others(cs):
                if (d := np.linalg.norm(cs.position - o.position)) <= 1.0:
                    # self.logger.info("Collision detected between craft %s at %s and "
                    #                      "craft %s at %s. They were %sm apart",
                    #                      cs.id, cs.position,
                    #                      o.id, o.position,
                    #                      d)
                    # raise BennuParticleReturn.CollisionEvent
                    pass

    def _send_cs_docked_msg(self, id: str):
        # Notify all other cubesats that a cubesat has docked
        mship = self.single_mothership
        for cs_id in self.cubesats:
            if cs_id == id:
                continue
            msg = msgs.CubeSatDocked(tx_id = mship.id,
                                     rx_id = cs_id,
                                     timestamp = self.simtime,
                                     id = id)
            self._comms_manager.send_msg(msgs.Message(msg))

    def _calc_reward(self):
        # Calculate the reward for this step.
        #
        # The reward is intended to encourage agents to get closer
        # to the mothership.
        #
        # Each reward calculation must be between 0 and 1
        closer = 0
        total = 0
        new_dists = self._calc_dists_to_mothership()
        for i, dist in new_dists.items():
            if i in self.distances_to_mothership:
                total += 1
                if dist < self.distances_to_mothership[i]:
                    closer += 1
        self.distances_to_mothership = new_dists
        if total == 0:
            assert len(self.cubesats) == 0
            reward = 0.0
        else:
            further = self.num_cubesats - closer
            reward = (0.7 * closer - 0.3 * further)/total
        assert (reward <= 1.0) and (reward >= -1.0)
        self.episode_reward += reward
        self.num_rewards_calculated += 1
            

    def _calc_dists_to_mothership(self):
        # Calculate all distances to the mothership and return as a dict: id -> dist
        mothership = self.single_mothership
        return {i: np.linalg.norm(c.position - mothership.position)
                for i, c in self.cubesats.items()}
        
    @property
    def single_mothership(self):
        """Return the only mothership in the simulation.

        This also checks and validates that there is only
        one mothership as expected.
        """
        if self.num_motherships != 1:
            raise ValueError("There can only be one mothership")
        return list(self.motherships.values())[0]

    @property
    def final_episode_reward(self):
        # Normalizes the episode reward by the number of iterations
        reward = self.episode_reward/self.num_rewards_calculated
        assert (reward >= -1.0) and (reward <= 1.0)
        return reward


def setup():
    """Setup the experiment.

    Configure and return an experiment object.
    """
    import sys
    import logging.config

    config: Config = {
        'bennu_pos': np.array([50, 0, 0]),
        'particle_database': None,
        'transmission_freq': 1.0/2.0,
        'action_space_rate': 1.0,
        'action_space_dock_dist': 10.0,
        'action_space_waypoint_dist': 2.5,
        'model': None,
        'num_iters_calc_reward': 10,
        'min_num_samples': 3,
        'max_num_samples': 15,
        'simulation_config': {
            'timestep': 0.5,
            'save_history': True,
            'mothership_config': [
                {
                    'initial_position': np.array([0, 0, 0], dtype=float),
                    'cubesat_capacity': np.random.randint(3, 15),
                    'fuel_capacity': None,  # unlimited fuel
                },
            ],
            'cubesat_config': [
                {
                    'fuel_capacity': 170,
                    'sample_capture_prob': 0.85
                 },
            ]
        }
    }

    logger_name = "BennuParticleReturnExperiment"
    logging_config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(filename)s %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d - %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
                "stream": sys.stdout,
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "level": "DEBUG",
                "filename": "experiment.log",
                "mode": "w",
            },
        },
        "loggers": {
            logger_name: {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(logger_name)

    return BennuParticleReturn(config, logger)

def run_debug():  # noqa D
    from DSRC.simulation import animate_simulation

    import pdb, traceback, sys

    try:
        # animate_simulation(setup().run())
        setup().run()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



if __name__ == '__main__':
    run_debug()
