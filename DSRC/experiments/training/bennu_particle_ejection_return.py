"""Training interface to Bennu Particle Ejection Sample Return experiment.

This training interface utilizes PyGAD to optimize a PyTorch model via a genetic
optimization process. The model is used as a Deep Q Network for the reinforcement
learning task of determining docking order.
"""


from DSRC.experiments import BennuParticleReturn
from DSRC.experiments.models.bennu_particle_ejection_return import Model

import pygad
from pygad import torchga
import numpy as np
import sys
import logging
import logging.config
import torch


# The PyGAD GA class checks that the fitness function accepts two arguments
# but uses the __code__ method which I can't figure how to reproduce in a partial.
# To make things more difficult the multiprocessing is pickling everything so we
# cannot use lambdas or local closures... So we spoof PyGAD with this hackery
# you see below. The optimizer class thinks that it gets some function object
# that accepts two arguments since we set the __code__ object to have some field
# co_argcount equal to 2
class hacky:
    def __init__(self):
        self.co_argcount = 2

class FitnessFunc:
    def __init__(self, trainer):
        self.f = trainer.fitness_func
        self.__code__ = hacky()
    def __call__(self, sol, idx):
        return self.f(sol, idx)


def on_generation(ga_instance):  # noqa D
    global save_dir
    
    # assert ga_instance.generations_completed == len(ga_instance.best_solutions_fitness)
    print(f"Generation             = {ga_instance.generations_completed}")
    print(f"Generation Max Fitness = {ga_instance.best_solutions_fitness[-1]}")
    print(f"Overall Max Fitness    = {max(ga_instance.best_solutions_fitness)}")
    print("="*45)

    if (ngens := ga_instance.generations_completed) % 5 == 0 or ngens == 1:
        ga_instance.save(f"/{save_dir}/ga-generation-{ngens}")

        
class Trainer:
    """Trainer class.

    Encapsulates the training of the reinforcement learning process.
    """

    MIN_FITNESS = 0.0

    def __init__(self, num_solutions: int, num_experiments_per_fitness: int,
                 checkpoint_fname: str=None):
        """Construct and initialize training interface.

        The training interface creats a PyGAD GA instance and a
        PyGAD TorchGA model instance. Each model is evaluated 
        using a full episode where an episode consists of one 
        BennuParticleReturn experiment.

        A model's fitness is calculated using
        num_experiments_per_fitness experiments averaged.
        """
        self.num_experiments_per_fitness = num_experiments_per_fitness

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        if checkpoint_fname is None:
            self.ga_instace = self._new_ga_instance(num_solutions)
        else:
            self.ga_instance = pygad.load(checkpoint_fname)

        self.experiment_config = {
            'bennu_pos': np.array([50, 0, 0]),
            'particle_database': None,
            'transmission_freq': 1.0/2.0,
            'action_space_rate': 1.0,
            'action_space_dock_dist': 10.0,
            'action_space_waypoint_dist': 2.5,
            'num_iters_calc_reward': 10,
            'simulation_config': {
                'timestep': 0.5,
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

        logger_name = "BennuParticleReturnTraining"
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
                    "level": "WARN",
                    "handlers": ["console"], #["console", "file"],
                    "propagate": False,
                },
            },
        }

        logging.config.dictConfig(logging_config)
        self.logger = logging.getLogger(logger_name)

    def fitness_func(self, sol, sol_idx) -> float:
        """Return the fitness of this model."""
        model = Model()
        weight_dict = torchga.model_weights_as_dict(model, sol)
        model.load_state_dict(weight_dict)
        model.to(self.device)

        config = self.experiment_config.copy()
        config['model'] = model

        fitness = [self._run_single_experiment(config)
                   for _ in range(self.num_experiments_per_fitness)]
        return sum(fitness)/self.num_experiments_per_fitness

    def _run_single_experiment(self, config):
        experiment = BennuParticleReturn(config, self.logger)

        try:
            experiment.run()
        except BennuParticleReturn.CollisionEvent:
            # In the event that a crash occured the fitness is set low
            return -2.0

        def assert_btwn(x, a, b):
            if not (x >= a) and (x <= b):
                print(f"ERROR: {x} should be between {a} and {b}")
                assert False

        # Return the fitness for a completed run
        p_cubesats_recovered = experiment.num_cubesats_recovered/experiment.init_num_cubesats
        assert_btwn(p_cubesats_recovered, 0.0, 1.0)
        if experiment.max_sample_value > 0.0:
            p_sample_value_recovered = experiment.sample_value_recovered/experiment.max_sample_value
            assert_btwn(p_sample_value_recovered, 0.0, 1.0)
        else:
            p_sample_value_recovered = 0.0
        p_invalid_docking_cmds = experiment.num_invalid_docking_commands/experiment.total_num_actions
        assert_btwn(p_invalid_docking_cmds, 0.0, 1.0)
        
        global_fitness = (0.65 * p_sample_value_recovered + 0.35 * p_cubesats_recovered) * (1.0 - p_invalid_docking_cmds)
        assert_btwn(global_fitness, 0.0, 1.0)

        fitness = 0.9 * global_fitness + 0.1 * experiment.final_episode_reward
        assert_btwn(fitness, -1.0, 1.0)

        return fitness

    def _new_ga_instance(self, num_solutions):
        model_ga = torchga.TorchGA(Model(), num_solutions)
        
        num_generations = 1500  # Number of generations
        num_parents_mating = 2  # Number of solutions to be selected as parents in the mating pool.
        initial_population = model_ga.population_weights  # Initial population of network weights
        parent_selection_type = "sss"  # Type of parent selection.
        crossover_type = "single_point"  # Type of the crossover operator.
        mutation_type = "random"  # Type of the mutation operator.
        mutation_percent_genes = 25  # Percentage of genes to mutate.
        # This parameter has no action if the parameter mutation_num_genes exists.
        keep_parents = 2  # Number of parents to keep in the next population.
        # -1 means keep all parents and 0 means keep nothing.

        fitness_func_wrapper = FitnessFunc(self)

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    initial_population=initial_population,
                                    fitness_func=fitness_func_wrapper,
                                    parent_selection_type=parent_selection_type,
                                    crossover_type=crossover_type,
                                    mutation_type=mutation_type,
                                    mutation_percent_genes=mutation_percent_genes,
                                    keep_parents=keep_parents,
                                    on_generation=on_generation,
                                    parallel_processing=["process", None])

    def run(self):
        """Run the training session.

        This just dispatches to the GA instance's run method.
        """
        self.ga_instance.run()
        self.ga_instance.save("/tmp/ga_results-final")
        #self.ga_instance.plot_result(title="Bennu Sample Return Fitness", linewidth=4)

        model = Model()
        sol, _, _ = self.ga_instance.best_solution()
        weight_dict = torchga.model_weights_as_dict(model, sol)
        model.load_state_dict(weight_dict)
        model.save("/tmp/bennu_particle_return_model_weights")


if __name__ == '__main__':
    import argparse
    import os
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_solutions",
                        type=int,
                        default=os.cpu_count())
    parser.add_argument("--num_experiments_per_fitness",
                        type=int,
                        default=2)
    parser.add_argument("--save_dir",
                        type=str,
                        default=None)
    parser.add_argument("--checkpoint_fname",
                        type=str,
                        default=None)

    args = parser.parse_args()

    if args.save_dir is None:
        from datetime import datetime
        args.save_dir = datetime.now().strftime("%B-%d-%H:%M:%S")
    
    #trainer = Trainer(num_solutions=8, num_experiments_per_fitness=1, checkpoint_fname="/tmp/ga-generation-65")
    trainer = Trainer(num_solutions=args.num_solutions,
                      num_experiments_per_fitness=args.num_experiments_per_fitness,
                      checkpoint_fname=args.checkpoint_fname)
    save_dir = args.save_dir  # Global accessed in the save function
    p = pathlib.Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    print("=" * 45)
    print("Training startiong....")
    print("=" * 45)
    trainer.run()
    
    # try:
    #     trainer.run()
    # except:
    #     import pdb, traceback, sys
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
