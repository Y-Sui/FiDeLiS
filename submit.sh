#!/bin/bash

#SBATCH --job-name=rog_test_cases          # Job name
#SBATCH --output=slurm_output.txt             # Output file
#SBATCH --error=slurm_error_log.txt           # Error log file
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --time=0:30:00                 # Time limit hrs:min:sec
#SBATCH --partition=standard           #standard, medium, and long
#SBATCH --gpus=a100:2			# Request two GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuan.sui@u.nus.edu

# Adjust the path to where conda is installed
source ~/.bashrc

conda activate ~/myenv/rog            # Activate your Conda environment

export HUGGING_FACE_HUB_TOKEN='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'
echo "Starting job on 2 A100 GPUs in the Conda environment"
# For example, run a Python script
bash ./scripts/planning.sh

# Deactivate Conda environment after the job is done
conda deactivate

