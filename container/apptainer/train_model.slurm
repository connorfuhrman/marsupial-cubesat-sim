#!/bin/bash

#SBATCH --job-name=dsrc_training
#SBATCH --ntasks=94
#SBATCH --mem-per-cpu=5gb
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --partition=standard
#SBATCH --account=jekan
#SBATCH --mail-type=ALL
#SBATCH --exclusive

apptainer run \
	  --bind apptainer_results:/mnt/train_output \
	  $HOME/spacetrex/DSRC/dsrc.sif \
	  python3 /app/DSRC/experiments/training/bennu_particle_ejection_return.py \
	  --num_experiments_per_fitness=2 \
	  --num_opt_procs=94 \
	  --num_solutions=188 \
	  --save_dir=/mnt/train_output > apptainer_results/run_output.log
