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
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
        print("="*45)
        
class Trainer:
    """Trainer class.

    Encapsulates the training of the reinforcement learning process.
    """

    MIN_FITNESS = 0.0

    def __init__(self, num_solutions: int):
        """Construct and initialize training interface.

        The training interface creats a PyGAD GA instance and a
        PyGAD TorchGA model instance. Each model is evaluated 
        using a full episode where an episode consists of one 
        BennuParticleReturn experiment.
        """
        model_ga = torchga.TorchGA(Model(), num_solutions)

        num_generations = 250  # Number of generations.
        num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
        initial_population = model_ga.population_weights  # Initial population of network weights
        parent_selection_type = "sss"  # Type of parent selection.
        crossover_type = "single_point"  # Type of the crossover operator.
        mutation_type = "random"  # Type of the mutation operator.
        mutation_percent_genes = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
        keep_parents = -1  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

        fitness_func_wrapper = FitnessFunc(self)

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    initial_population=initial_population,
                                    # fitness_func=lambda sol, idx: self.fitness_func(sol, idx),
                                    fitness_func=fitness_func_wrapper,
                                    parent_selection_type=parent_selection_type,
                                    crossover_type=crossover_type,
                                    mutation_type=mutation_type,
                                    mutation_percent_genes=mutation_percent_genes,
                                    keep_parents=keep_parents,
                                    on_generation=on_generation,
                                    parallel_processing=["process", None])

        self.experiment_config = {
            'bennu_pos': np.array([15, 0, 0]),
            'particle_database': None,
            'transmission_freq': 1.0/5.0,
            'action_space_rate': 5.0,
            'action_space_dock_dist': 10.0,
            'action_space_waypoint_dist': 5.0,
            'simulation_config': {
                'timestep': 0.5,
                'mothership_config': [
                    {
                        'initial_position': np.array([0, 0, 0], dtype=float),
                        'cubesat_capacity': np.random.randint(3, 15),
                        'fuel_capacity': None,  # unlimited fuel
                    },
                ],
                'cubesat_config': [
                    {
                        'fuel_capacity': 1e6,  # practically unlimited fuel
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
                    "handlers": [], #["console", "file"],
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

        config = self.experiment_config.copy()
        config['model'] = model

        experiment = BennuParticleReturn(config, self.logger)

        try:
            experiment.run()
        except BennuParticleReturn.CollisionEvent:
            # In the event that a crash occured the fitness is set low
            return 0.0

        # Return the fitness for a completed run
        # TODO update this
        fitness = experiment.num_cubesats_recovered/experiment.init_num_cubesats

        del experiment

        return fitness

    def run(self):
        """Run the training session.

        This just dispatches to the GA instance's run method.
        """
        self.ga_instance.run()
        self.ga_instance.plot_result(title="Bennu Sample Return Fitness", linewidth=4)

        model = Model()
        sol, _, _ = self.ga_instance.best_solution()
        weight_dict = torchga.model_weights_as_dict(model, sol)
        model.load_state_dict(weight_dict)
        model.save("bennu_particle_return_model_weights")


if __name__ == '__main__':
    trainer = Trainer(num_solutions=15)
    try:
        trainer.run()
    except:
        import pdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
