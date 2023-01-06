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
import os
import multiprocessing as mp
import itertools as it
from torch.utils.tensorboard import SummaryWriter

def fitness_func(sol, sol_idx):
    global trainer

    return trainer.fitness_func(sol, sol_idx)

def on_generation(ga_instance: pygad.GA):  # noqa D
    global save_dir
    global summary_writer
    global stop_fitness
    global max_fitness

    max_last_gen_fitness = max(ga_instance.last_generation_fitness)
    if max_fitness is None or max_fitness < max_last_gen_fitness:
        max_fitness = max_last_gen_fitness
    min_last_gen_fitness = min(ga_instance.last_generation_fitness)
    avg_last_gen_fitness = np.average(ga_instance.last_generation_fitness)
    
    # assert ga_instance.generations_completed == len(ga_instance.best_solutions_fitness)
    N = ga_instance.generations_completed
    print(f"Generation             = {N}")
    print(f"Generation Max Fitness = {max_last_gen_fitness}")
    print(f"Generation Min Fitness = {min_last_gen_fitness}")
    print(f"Generation Avg Fitness = {avg_last_gen_fitness}")
    print(f"Overall Max Fitness    = {max_fitness}")
    print("="*45)

    summary_writer.add_scalar("Generation", N, N)
    summary_writer.add_scalar("Fitness/Max Overall", max_fitness, N)
    summary_writer.add_scalar("Fitness/Generation Max", max_last_gen_fitness, N)
    summary_writer.add_scalar("Fitness/Generation Min", min_last_gen_fitness, N)
    summary_writer.add_scalar("Fitness/Generation Avg", avg_last_gen_fitness, N)
    summary_writer.add_histogram("Fitness/Generation", ga_instance.last_generation_fitness, ga_instance.generations_completed)
    summary_writer.flush()

    with open(f"{save_dir}/fitness-values.csv", "a") as f:
        vals = ",".join(map(str, ga_instance.last_generation_fitness)) + "\n"
        f.write(vals)
    
    if (ngens := ga_instance.generations_completed) % 5 == 0 or ngens == 1:
        ga_instance.save(f"{save_dir}/ga-generation-{ngens}")

    if max_fitness > stop_fitness:
        print(f"Signaling to stop since fitness reached {stop_fitness} at generation {N}")
        return "stop"

        
class Trainer:
    """Trainer class.

    Encapsulates the training of the reinforcement learning process.
    """

    MIN_FITNESS = 0.0

    def __init__(self, config,
                 num_solutions: int,
                 num_opt_procs: int,
                 initial_model = None,
                 num_experiments_per_fitness: int=1,
                 num_proc_per_fitness: int=1,
                 checkpoint_fname: str=None):
        """Construct and initialize training interface.

        The training interface creats a PyGAD GA instance and a
        PyGAD TorchGA model instance. Each model is evaluated 
        using a full episode where an episode consists of one 
        BennuParticleReturn experiment.

        A model's fitness is calculated using
        num_experiments_per_fitness experiments averaged.
        """
        self.experiment_config = config
        
        self.num_experiments_per_fitness = num_experiments_per_fitness
        self.num_proc_per_fitness = num_proc_per_fitness
        self.num_opt_procs = num_opt_procs
        self.initial_model = initial_model

        print(f"Training with population size of {num_solutions}")
        if self.initial_model is not None:
            print("This training run was seeded with existing model weights")
        print(f"Fitness is averaged out of {self.num_experiments_per_fitness} runs")
        print(f"Fitness is calculated using {self.num_proc_per_fitness} procs")
        print(f"Optimizer uses {os.cpu_count() if (n := self.num_opt_procs) is None else n} procs")
        print(f"Number of available processor cores is {os.cpu_count()}")
        print("="*45)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        if checkpoint_fname is None:
            self.ga_instace = self._new_ga_instance(num_solutions)
        else:
            self.ga_instance = pygad.load(checkpoint_fname)

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
                    "level": "CRITICAL",
                    "stream": sys.stdout,
                },
                # "file": {
                #     "class": "logging.FileHandler",
                #     "formatter": "standard",
                #     "level": "DEBUG",
                #     "filename": "experiment.log",
                #     "mode": "w",
                # },
            },
            "loggers": {
                logger_name: {
                    "level": "CRITICAL",
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

        fitness = [self._run_single_experiment(config.copy())
                   for _ in range(self.num_experiments_per_fitness)]

        # if self.num_opt_procs == 1:
        #     print("Calculating!")
        #     fitness = [self._run_single_experiment(config)
        #                for _ in range(self.num_experiments_per_fitness)]
        # else:
        #     # print(f"Opening mp pool with {self.num_proc_per_fitness} procs")
        #     with mp.Pool(self.num_opt_procs) as pool:
        #         fitness = pool.map(self._run_single_experiment, it.repeat(config, self.num_experiments_per_fitness))
        #     # print("Calculated fitness!")

        avg_fitness = sum(fitness)/self.num_experiments_per_fitness
        if type(avg_fitness) not in pygad.GA.supported_float_types:
            print(f"Type avg_fitness {avg_fitness} is {type(avg_fitness)}")
            raise RuntimeError()
        return avg_fitness

    def _run_single_experiment(self, config):
        # Chose a random location for the mothership
        mship_pos = np.random.default_rng().uniform(low=-500, high=500, size=(3,))
        # Chose a random Bennu position some distance away
        theta = np.random.default_rng().uniform(low=0, high=2.0*np.pi)
        phi = np.random.default_rng().uniform(low=0, high=np.pi)
        bennu_pos = np.array([dist_from_bennu * np.cos(theta) * np.sin(phi),
                              dist_from_bennu * np.sin(theta) * np.sin(phi),
                              dist_from_bennu * np.cos(phi)], dtype=float) + mship_pos

        config['bennu_pos'] = bennu_pos
        config['simulation_config']['mothership_config'][0]['initial_position'] = mship_pos

        assert np.abs(np.linalg.norm(bennu_pos - mship_pos) - dist_from_bennu) <= 0.5  # Sanity check the distance from the mothership

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

        if p_sample_value_recovered == 0.0 and p_cubesats_recovered == 0.0:
            global_fitness = -1.0
        else:
            global_fitness = (0.65 * p_sample_value_recovered + 0.35 * p_cubesats_recovered) * (1.0 - p_invalid_docking_cmds)
            assert_btwn(global_fitness, 0.0, 1.0)

        fitness = (0.9 * global_fitness) + (0.1 * experiment.final_episode_reward)
        assert_btwn(fitness, -1.0, 1.0)

        return fitness

    def _new_ga_instance(self, num_solutions):
        if self.initial_model is None:
            initial_population = torchga.TorchGA(Model(), num_solutions).population_weights
        else:
            initial_population = torchga.TorchGA(self.initial_model, num_solutions).population_weights
        
        num_generations = 1500  # Number of generations
        num_parents_mating = num_solutions  # Number of solutions to be selected as parents in the mating pool.
        parent_selection_type = "sss"  # Type of parent selection.
        crossover_type = "single_point"  # Type of the crossover operator.
        mutation_type = "random"  # Type of the mutation operator.
        mutation_percent_genes = 25  # Percentage of genes to mutate.

        parallel = ["process", n] if (n := self.num_opt_procs) is None or n > 1 else None

        self.ga_instance = pygad.GA(num_generations=num_generations,
                                    num_parents_mating=num_parents_mating,
                                    initial_population=initial_population,
                                    fitness_func=fitness_func,
                                    parent_selection_type=parent_selection_type,
                                    crossover_type=crossover_type,
                                    mutation_type=mutation_type,
                                    mutation_percent_genes=mutation_percent_genes,
                                    on_generation=on_generation,
                                    parallel_processing=parallel,
                                    allow_duplicate_genes=False,
                                    keep_elitism=round(num_solutions/4))

    def run(self):
        """Run the training session.

        This just dispatches to the GA instance's run method.
        """
        global save_dir

        self.ga_instance.run()
        self.ga_instance.save(f"{save_dir}/ga_results-final")
        #self.ga_instance.plot_result(title="Bennu Sample Return Fitness", linewidth=4)

        model = Model()
        sol, fitness, _ = self.ga_instance.best_solution()
        weight_dict = torchga.model_weights_as_dict(model, sol)
        model.load_state_dict(weight_dict)
        return model, fitness


if __name__ == '__main__':
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_solutions",
                        type=int,
                        default=os.cpu_count())
    parser.add_argument("--num_opt_procs",
                        type=int,
                        default=os.cpu_count())
    parser.add_argument("--num_experiments_per_fitness",
                        type=int,
                        default=1)
    # parser.add_argument("--num_procs_per_fitness",
    #                      type=int,
    #                      default=1)
    parser.add_argument("--save_dir",
                        type=str,
                        default=None)
    # parser.add_argument("--checkpoint_fname",
    #                     type=str,
    #                     default=None)

    args = parser.parse_args()

    if args.save_dir is None:
        from datetime import datetime
        args.save_dir = datetime.now().strftime("%B-%d-%H:%M:%S")
    
    #trainer = Trainer(num_solutions=8, num_experiments_per_fitness=1, checkpoint_fname="/tmp/ga-generation-65")
    experiment_config = {
        'bennu_pos': np.array([50, 0, 0]),
        'particle_database': None,
        'transmission_freq': 1.0/2.0,
        'action_space_rate': 1.0,
        'action_space_dock_dist': 10.0,
        'action_space_waypoint_dist': 2.5,
        'num_iters_calc_reward': 10,
        'min_num_samples': 2,
        'max_num_samples': 5,
        'simulation_config': {
            'timestep': 0.5,
            'save_history': False,
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

    dist_from_bennu = 50
    num_samples = [3, 4, 5, 10, 15, 20, 50]
    starting_model = None
    for ns in num_samples:
        print("=" * 45)
        print(f"Training starting with {ns} samples ....")
        print("=" * 45)

        experiment_config['max_num_samples'] = ns
        trainer = Trainer(config=experiment_config,
                          initial_model=starting_model,
                          num_solutions=args.num_solutions,
                          num_opt_procs=args.num_opt_procs,
                          num_experiments_per_fitness=args.num_experiments_per_fitness,
                          num_proc_per_fitness=1)
                          # checkpoint_fname=args.checkpoint_fname)
        #################
        # Global Access #
        save_dir = pathlib.Path(args.save_dir) / f"{ns}-samples"
        summary_writer = SummaryWriter(save_dir)
        max_fitness = None
        stop_fitness = 0.75 if ns != num_samples[-1] else 0.9
        ################
    
        save_dir.mkdir(parents=True, exist_ok=True)

        # try:
        #     initial_pop, last_fitness = trainer.run()
        # except KeyboardInterrupt:
        #     raise
        # except:
        #     import pdb, traceback, sys
        #     extype, value, tb = sys.exc_info()
        #     traceback.print_exc()
        #     pdb.post_mortem(tb)
        starting_model, last_fitness = trainer.run()
        torch.save(starting_model.state_dict(), save_dir/"trained.pytorch_model")
        print(f"Training with {ns} samples finished with fitness {last_fitness}")
        print("=" * 45)
    
    # try:
    #     trainer.run()
    # except:
    #     import pdb, traceback, sys
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
