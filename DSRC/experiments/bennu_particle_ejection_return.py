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

import logging
import numpy as np
import pathlib as pl
import pickle
import random
from typing import TypedDict


class Config(TypedDict):
    """Configuration for the Bennu Particle Return experiment."""

    simulation_config: SimulationConfig
    """Basic simulation config."""

    bennu_pos: np.ndarray
    """Position of Bennu in the simulation."""

    particle_database: pl.Path
    """Path to pickled database of particle ejections."""


class OutOfData(Exception):  # noqa D
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
        pos_from_bennu = lambda: np.random.uniform(10.0, 20.0, size=(3,))
        samples = [Sample(0.0, np.random.uniform(0.0, 10.0),
                          (pos_from_bennu() - self.bennu_orig).copy(),
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


class BennuParticleReturn(Simulation):
    """Bennu Particle Ejection Return Expriment.

    The experiment simulates the return of some number of cubestats after
    the cubesats captured ejected particles from the aseroid Bennu.
    """

    def __init__(self, config: Config, logger: logging.Logger = None):  # noqa D
        if logger is None:
            # Assume this is made within a Ray actor
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger()
        self.logger = logger

        super().__init__(config['simulation_config'], self.logger)

        self.logger.info("Reading from database at %s", config['particle_database'])
        self.particle_data = ParticleDatabase(config['particle_database'],
                                              config['bennu_pos'],
                                              self.logger)


        self._initialize()

    ################################################
    # Override methods required in base simulation #
    ################################################
    def _planning_step(self):
        mothership = self.single_mothership
        for c in self.cubesats.values():
            c.add_waypoint(mothership.position)
            if np.linalg.norm(c.position - mothership.position) <= 0.5:
                mothership.dock_cubesat(c)
                del self._crafts[c.id]

    def _is_terminated(self):
        """Determine termination event.

        Simulation is terminated when there are no more cubesats.
        """
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
        # Draw between 5 and 15 samples
        n_samps = int(np.random.uniform(low=5, high=15))
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
            cs.add_waypoint(self.single_mothership.position)
            self._crafts[cs.id] = cs
            self.logger.info("Cubesat %s is starting at %s "
                              "and has %s it's sample with value %s",
                              cs.id,
                              cs.position,
                              ("" if cs.has_sample else "not ") + "captured",
                              s.value)

    @property
    def single_mothership(self):
        """Return the only mothership in the simulation.

        This also checks and validates that there is only
        one mothership as expected.
        """
        if self.num_motherships != 1:
            raise ValueError("There can only be one mothership")
        return list(self.motherships.values())[0]


def setup():
    """Setup the experiment.

    Configure and return an experiment object.
    """

    import sys
    import logging.config

    config: Config = {
        'bennu_pos': np.array([10, 0, 0]),
        'particle_database': None,
        'simulation_config': {
            'timestep': 0.5,
            'mothership_config': [
                {
                    'initial_position': np.array([0, 0, 0], dtype=float),
                    'cubesat_capacity': np.random.randint(3, 15),
                    'fuel_capacity': 100,
                },
            ],
            'cubesat_config': [
                {
                    'fuel_capacity': 10,
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
                "format": "%(asctime)s %(levelname)s: %(message)s",
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
        animate_simulation(setup().run())
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



if __name__ == '__main__':
    run_debug()
