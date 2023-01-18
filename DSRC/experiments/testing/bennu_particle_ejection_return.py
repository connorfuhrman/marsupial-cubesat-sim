"""Test trained model on the Bennu sample return experiment."""

import torch
torch.set_grad_enabled(False)

from DSRC.experiments import BennuParticleReturn
from DSRC.experiments.models.bennu_particle_ejection_return import Model
from DSRC.simulation.utils import save_json_file
from DSRC.simulation import animate_simulation

from pathlib import Path
import numpy as np



def load_model(file: Path) -> Model:
    if not file.exists() or not file.is_file():
        raise RuntimeError(f"{file} is either not found or isn't a file")

    return torch.load(file)


def run_experiment(saved_model: Path):
    if type(saved_model) is str:
        saved_model = Path(saved_model)

    model = Model()
    model.load_state_dict(load_model(saved_model))

    experiment_config = {
        'bennu_pos': np.array([50, 0, 0]),
        'particle_database': None,
        'transmission_freq': 2.0,
        'action_space_rate': 1.0,
        'action_space_dock_dist': 25.0,
        'action_space_waypoint_dist': 2.5,
        'num_iters_calc_reward': 4,
        'min_num_samples': 5,
        'max_num_samples': 10,
        'model': model,
        'simulation_config': {
            'timestep': 0.25,
            'save_history': True,
            'mothership_config': [
                {
                    'initial_position': np.array([5, 5, 0], dtype=float),
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

    try:
        experiment = BennuParticleReturn(experiment_config)
        history = experiment.run()
        save_path = Path("./testing_results")
        save_path.mkdir(exist_ok=True)
        save_json_file(history, save_path/f"sim_history-{experiment.id}.json")
        animate_simulation(history)
    except BennuParticleReturn.CollisionEvent:
        print("ERROR: Test ended in a collision")
        exit(1)
    
        
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", "-m",
                        type=str,
                        help="Path to saved model to run")

    args = parser.parse_args()

    run_experiment(args.model)
