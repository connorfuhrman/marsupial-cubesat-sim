"""Experiment to simulate cubesats capturing their closest sample."""

from DSRC.simulation import (
    Simulation,
    SimulationConfig,
    CubeSatConfig,
    MothershipConfig,
)
from DSRC.simulation.spacecraft import Mothership, CubeSat
from DSRC.simulation.communication import messages as msgs
from DSRC.simulation.samples import Sample
import logging

import numpy as np


class Experiment(Simulation):
    """Instantiation of the DSRC simulation class.

    This experiment sends commands from the mothership
    to the deployed cubesats to capture the sample that's
    closest to them.
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        max_num_samples: int,
        timeout: float,
        logger: logging.Logger = None,
    ):
        self._max_num_samples = max_num_samples
        self._timeout = timeout
        self._cs_sample_assignments = dict()
        if logger is None:
            # Assume this is made within a Ray actor
            logging.basicConfig(level=logging.ERROR)
            self._logger = logging.getLogger()
        else:
            self._logger = logger

        super().__init__(sim_config, self._logger)

    ############################################
    # Override the required methdos for this ABC
    ############################################

    def _is_terminated(self) -> bool:
        """Stop the sim on condition.

        - If it's been less than 1 min sim time don't stop
        - If we've timed out then stop
        - If there are no cubesats left then stop bc all have docked
        """
        if self.simtime < 60:
            return False
        if self.simtime > self._timeout:
            self._logger.warning("Timeout condition reached")
            return True
        if self.num_cubesats == 0:
            self._logger.info("All cubesats docked. Signaling sim to end.")
            return True
        return False

    def _planning_step(self):
        self._do_cubesat_msg_cb()
        self._do_cubesat_captures()
        mothership = list(self.motherships.values())[0]
        if self.iter == 0:
            self._on_first_iter()
        elif self.num_samples > 0:
            self._do_sample_assignment(mothership)
        elif self.iter > 100:
            for c in self.cubesats.values():
                c.add_waypoint(mothership.position)
                if np.linalg.norm(c.position - mothership.position) <= 0.5:
                    mothership.dock_cubesat(c)
                    del self._crafts[c.id]

    def _update_samples(self):
        if (
            np.random.uniform() > 0.5
            and self.num_samples < self._max_num_samples
            and self.simtime < 5 * 60
        ):
            nsamps = np.random.randint(3, 5)
            new_samps = [self._make_sample() for _ in range(nsamps)]
            self._logger.debug("Adding %s samples to the sim", nsamps)
            self._samples.extend(new_samps)

    ########################################
    # Helper methods
    ########################################

    def _on_first_iter(self):
        if self.num_motherships > 1:
            self._logger.warn(
                "There are %s motherships but this is really only set "
                "up for 1. There will probably be an error somewhere",
                self.num_motherships,
            )
        for m in self.motherships.values():
            while m.can_deploy_cubesat:
                self._do_cubesat_deploy(m)

    def _do_cubesat_deploy(self, mship):
        offset = np.array([0, 0, 0.5 * (self.num_cubesats + 1)], dtype=float)
        deploy_pos = mship.position - offset
        to_deploy = CubeSat(
            deploy_pos.copy(),
            self._cubesat_configs[mship.id]["fuel_capacity"],
            0.95,
            self._logger,
        )
        mship.deploy_cubesat()
        self._crafts[to_deploy.id] = to_deploy

    def _make_sample(self):
        return Sample(
            weight=np.random.uniform(5, 10),
            value=np.random.uniform(1, 10),
            position=np.random.uniform(-15, 15, 3),
            velocity=np.zeros(3),
        )

    def _do_sample_assignment(self, mship):
        if self.num_samples == 0:
            return
        for c in filter(
            lambda c: c not in self._cs_sample_assignments.values(),
            self.cubesats.values(),
        ):
            samps = filter(lambda s: s not in self._cs_sample_assignments, self.samples)
            samps = [s for s in samps]
            dists = [np.linalg.norm(c.position - s.position) for s in samps]
            if len(dists) == 0:
                return
            s = samps[np.argmin(dists)]
            self._cs_sample_assignments[s] = c
            msg = msgs.SampleAquireCommand(
                tx_id=mship.id,
                rx_id=c.id,
                timestamp=self.simtime,
                sample_pos=s.position,
            )
            self._comms_manager.send_msg(msgs.Message(msg), mship, c)

    def _do_cubesat_msg_cb(self):  # noqa D
        for cs, tx_time, msg in self.cubesat_msg_iterator:
            if msgs.Message.is_type(msg, msgs.SampleAquireCommand):
                self._logger.debug(
                    "Cubesat %s was commanded to capture sample at %s",
                    cs.id,
                    msg.msg["sample_pos"],
                )
                cs.add_waypoint(msg.msg["sample_pos"].copy(), front=True)
            else:
                raise ValueError(f"Don't know how to handle message {msg}")

    def _do_cubesat_captures(self):  # noqa D
        captured = []
        for s, c in self._cs_sample_assignments.items():
            if np.linalg.norm(s.position - c.position) <= 1:
                self._logger.debug(
                    "Cubesat %s attempting capture at %s", c.id, s.position
                )
                if c.attempt_sample_capture(s):
                    self._logger.debug("Capture was succesful!!")
                    captured.append(s)
                    c.drop_curr_waypoint()

        for s in captured:
            del self._cs_sample_assignments[s]
            self._samples.remove(s)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from DSRC.ray import ActorPool
    from DSRC.simulation.utils import save_json_file
    import ray
    import logging
    import sys
    import json
    import pickle
    from pathlib import Path

    def get_sim_config():  # noqa D
        return {
            "mothership_config": [
                {
                    "initial_position": np.array([0, 0, 0], dtype=float),
                    "cubesat_capacity": np.random.randint(3, 15),
                    "fuel_capacity": 100,
                },
            ],
            "cubesat_config": [
                {"fuel_capacity": 10},
            ],
            "timestep": 0.1,
        }

    ExperimentActor = ray.remote(Experiment)

    def get_experiment_actor(max_samps):  # noqa D
        return

    def main():  # noqa D
        parser = ArgumentParser()
        parser.add_argument(
            "--max_samples",
            type=int,
            help="The max number of samples in a simulation at any one iteration.",
            default=15,
        )
        parser.add_argument(
            "--num_experiments", type=int, help="How many experiments to run", default=4
        )
        parser.add_argument(
            "--single_threaded", help="Don't use Ray", action="store_true"
        )
        parser.add_argument(
            "--no_animation", help="Don't render an animation", action="store_true"
        )
        parser.add_argument(
            "--no_save", help="Don't save results to file", action="store_true"
        )

        args = parser.parse_args()

        def get_actor():
            return ExperimentActor.remote(get_sim_config(), args.max_samples, 60 * 60)

        if not args.single_threaded:
            pool = ActorPool(get_actor, lambda a: a.run.remote(), args.num_experiments)
            res = pool.run()
        else:
            DEFAULT_LOGGING = {
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
                    "CaptureClosestSample": {
                        "level": "DEBUG",
                        "handlers": ["console", "file"],
                        "propagate": False,
                    },
                },
            }

            import logging.config

            logging.config.dictConfig(DEFAULT_LOGGING)
            logger = logging.getLogger("CaptureClosestSample")

            res = [
                Experiment(get_sim_config(), args.max_samples, 60 * 30, logger).run()
                for _ in range(args.num_experiments)
            ]

        if not args.no_animation:
            from DSRC.simulation import animate_simulation

            animate_simulation(res)

        if not args.no_save:
            for i, r in enumerate(res):
                save_path = Path(f"experiment_results/run_{i}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_json_file(r, save_path / "results.json")
                with open(save_path / "results.pkl", "wb") as f:
                    pickle.dump(r, f)

    main()
