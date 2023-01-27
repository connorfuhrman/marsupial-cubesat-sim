"""Expert system implementation of decentralized coordination for docking order."""

from DSRC.simulation.spacecraft import Mothership, CubeSat
from DSRC.experiments.bennu_particle_ejection_return import ObservationSpace
from DSRC.experiments import BennuParticleReturn
from DSRC.simulation.utils import save_json_file
from DSRC.simulation import animate_simulation
import multiprocessing as mp

import logging
import numpy as np
from functools import cmp_to_key
from functools import partial
from tqdm.contrib.concurrent import process_map

class ExpertSystem:
    def __init__(self, rate: float, min_dock_range: float, waypoint_dist: float,
                 craft_logger: logging.Logger):
        self.logger = logging.getLogger(craft_logger.name + ".action_space")
        self.docking_order = []
        self.starting_fuel_level = 175  # magic number from config
        self.is_docking = False
        self.period = 1.0/rate
        self.next_update = 0.0
        self.mship_position = None

    def update(self, craft: CubeSat, mship: Mothership, sim_time: float, model, obs: ObservationSpace):
        self.mship_position = mship.position
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

        # if sim_time < self.next_update and self.docking_order[0] != craft.id:
        #     return

        # self.next_update = sim_time + self.period
        if self.is_docking:
            # If we're moving to dock then do nothing
            return

        if self.docking_order[0] == craft.id:
            craft.clear_waypoints()
            craft.add_waypoint(mship.position)
            self.is_docking = True
        else:
            if craft.num_waypoints > 0:
                return
            # Move closer to the mothership
            theta = np.random.uniform(low=-1.0*np.pi, high=np.pi)
            phi = np.random.uniform(low=-1.0*np.pi/2.0, high=np.pi/2.0)
            r = np.random.uniform(low=10.0, high=15.0)

            wypnt = np.array([r * np.cos(theta) * np.sin(phi),
                              r * np.sin(theta) * np.sin(phi),
                              r * np.cos(phi)], dtype=float) + mship.position

            self.logger.debug("Moving to random position near mothership: %s",
                              wypnt)
            craft.add_waypoint(wypnt)

    def dist_to_mship(self, pos):
        return np.linalg.norm(pos - self.mship_position)
            

    def do_docking_order(self, craft: CubeSat, obs_space: ObservationSpace, sim_time):

        dists = [self.dist_to_mship(s['position']) for s in obs_space.state.values()]
        dists.append(self.dist_to_mship(craft.position))
        min_dist = min(dists)

        def proxy(craft_id):
            if craft_id == craft.id:
                state = craft.get_state_msg(sim_time)
            else:
                state = obs_space.state[craft_id]
            if state["fuel_level"] <= 25:
                # Preempt if running low on fuel
                return 0
            sample_cmpnt = 10.0 - state['sample_value']  # 10 is magic number but is max sample value possible
            dist_cmpnt = self.dist_to_mship(state['position']) - min_dist
            fuel_cmpnt = state['fuel_level']
            
            return sample_cmpnt + dist_cmpnt + fuel_cmpnt

        order = [i for i in obs_space.state.keys()]
        order.append(craft.id)
        self.docking_order = sorted(order, key=proxy)
        


def run_experiment(idx, experiment_config, logger):
    # print(f"Running experiment {idx}")
    experiment = BennuParticleReturn(experiment_config, ExpertSystem, logger)
    experiment.run()
    p_cubesats_recovered = experiment.num_cubesats_recovered/experiment.init_num_cubesats
    p_sample_recovered = experiment.sample_value_recovered/experiment.max_sample_value
    # print(f"Experiment {idx} concluded")
    return (p_cubesats_recovered, p_sample_recovered)


if __name__ == '__main__':
    import logging.config
    import sys
    import os
    
    experiment_config = {
        'bennu_pos': np.array([50, 0, 0]),
        'particle_database': None,
        'transmission_freq': 2.0,
        'action_space_rate': 1.0,
        'action_space_dock_dist': 25.0,
        'action_space_waypoint_dist': 2.5,
        'num_iters_calc_reward': 4,
        'min_num_samples': 5,
        'max_num_samples': 15,
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
                    'fuel_capacity': 250,
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



    num_expr = 8
    f = partial(run_experiment, experiment_config=experiment_config, logger=logger)
    # with mp.Pool(os.cpu_count()) as p:
    #     results = p.map(f, range(num_expr))
    results = process_map(f, range(num_expr), max_workers=os.cpu_count())

    print("Results:")
    for r in results:
        print(f"\t % cubesats recovered: {r[0]}. % sample value recovered: {r[1]}")


    # experiment = BennuParticleReturn(experiment_config, ExpertSystem, logger)
    # history = experiment.run()
    # # save_json_file(history, "./experiment_results/history.json")
    # animate_simulation([history])
