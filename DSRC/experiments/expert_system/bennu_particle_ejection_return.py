"""Expert system implementation of decentralized coordination for docking order."""

from DSRC.simulation.spacecraft import Mothership, CubeSat
from DSRC.experiments.bennu_particle_ejection_return import ObservationSpace
from DSRC.experiments import BennuParticleReturn
from DSRC.simulation.utils import save_json_file
from DSRC.simulation import animate_simulation

import logging
import numpy as np
from functools import cmp_to_key

class ExpertSystem:
    def __init__(self, rate: float, min_dock_range: float, waypoint_dist: float,
                 craft_logger: logging.Logger):
        self.logger = logging.getLogger(craft_logger.name + ".action_space")
        self.docking_order = []
        self.starting_fuel_level = 175  # magic number from config
        self.is_docking = False

    def update(self, craft: CubeSat, mship: Mothership, sim_time: float, model, obs: ObservationSpace):
        if craft.num_waypoints != 0:
            if len(self.docking_order) == 0:
                return
            elif self.docking_order[0] == craft.id and not self.is_docking:
                craft.clear_waypoints()
            else:
                return
        
        if len(obs.state) == 0 and sim_time <= 60:
            # If the cubesat hasn't gotten any messages yet in 30 seconds then we do nothing
            self.logger.info("No observations yet. Moving randomly")
            theta = np.random.default_rng().uniform(low=0, high=2.0*np.pi)
            phi = np.random.default_rng().uniform(low=0, high=np.pi)
            r = 0.5

            wypnt = np.array([r * np.cos(theta) * np.sin(phi),
                              r * np.sin(theta) * np.sin(phi),
                              r * np.cos(phi)], dtype=float) + craft.position
            craft.add_waypoint(wypnt)
            return
        
        self.do_docking_order(craft, obs, sim_time)
        self.logger.debug("Docking order is %s", self.docking_order)

        dist_to_mship = np.linalg.norm(craft.position - mship.position)

        if self.docking_order[0] == craft.id:
            # If we are first up then move midway to the mothership
            # unless we're already close enough
            # self.logger.info("We're first up in the docking order")
            if not self.is_docking:
                # self.logger.info("Moving to dock with mothership")
                craft.add_waypoint(mship.position)
                self.is_docking = True
        else:
            if np.random.default_rng().uniform(low=0, high=1) <= 0.3:
                # Move closer to the mothership
                # self.logger.info("Moving halfway to the mothership but am not first up in docking order")
                theta = np.random.default_rng().uniform(low=0, high=2.0*np.pi)
                phi = np.random.default_rng().uniform(low=0, high=np.pi)
                r = np.random.default_rng().uniform(low=3.0, high=7.0)

                wypnt = np.array([r * np.cos(theta) * np.sin(phi),
                                  r * np.sin(theta) * np.sin(phi),
                                  r * np.cos(phi)], dtype=float) + mship.position
                
                craft.add_waypoint(wypnt)
            else:
                # Move randomly closeby to gain situational awareness
                # self.logger.info("Moving randomly to gain SA")
                theta = np.random.default_rng().uniform(low=0, high=2.0*np.pi)
                phi = np.random.default_rng().uniform(low=0, high=np.pi)
                r = 0.15

                wypnt = np.array([r * np.cos(theta) * np.sin(phi),
                                  r * np.sin(theta) * np.sin(phi),
                                  r * np.cos(phi)], dtype=float) + craft.position
                craft.add_waypoint(wypnt)
            

    def do_docking_order(self, craft: CubeSat, obs_space: ObservationSpace, sim_time):

        def proxy(craft_id):
            if craft_id == craft.id:
                state = craft.get_state_msg(sim_time)
            else:
                state = obs_space.state[craft_id]
            if state["fuel_level"] <= 25:
                return 0
            
            return 10.0 - state['sample_value']

        order = [i for i in obs_space.state.keys()]
        order.append(craft.id)
        self.docking_order = sorted(order, key=proxy)
        


if __name__ == '__main__':
    import logging.config
    import sys
    
    experiment_config = {
        'bennu_pos': np.array([15, 15, 0]),
        'particle_database': None,
        'transmission_freq': 2.0,
        'action_space_rate': 1.0,
        'action_space_dock_dist': 25.0,
        'action_space_waypoint_dist': 2.5,
        'num_iters_calc_reward': 4,
        'min_num_samples': 3,
        'max_num_samples': 3,
        'model': "expert_system",
        'simulation_config': {
            'timestep': 0.25,
            'save_history': True,
            'mothership_config': [
                {
                    'initial_position': np.array([0, 0, 0], dtype=float),
                    'cubesat_capacity': 500,
                    'fuel_capacity': None,  # unlimited fuel
                },
            ],
            'cubesat_config': [
                {
                    'fuel_capacity': 175,
                    'sample_capture_prob': 0.85
                },
            ]
        }
    }

    logger_name = "BennuParticleReturnExperiment_ES"
    logging_config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(name)s %(levelname)s: %(message)s",
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
                "filename": "experiment_results/experiment.log",
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


    experiment = BennuParticleReturn(experiment_config, ExpertSystem, logger)
    history = experiment.run()
    save_json_file(history, "./experiment_results/history.json")
    animate_simulation(history)
