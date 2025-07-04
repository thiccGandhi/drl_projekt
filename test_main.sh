#!/bin/bash
#SBATCH --job-name=test_td3_impl            # Name des Jobs
#SBATCH --nodes=1                        # Anzahl der Nodes
#SBATCH --ntasks=1                       # Anzahl der Tasks
#SBATCH --cpus-per-task=1                # CPU-Kerne pro Task
#SBATCH --mem=2G                        # Arbeitsspeicher pro Task
#SBATCH --partition=cpu                  # Partition auf dem das ganze läuft
#SBATCH --time=48:00:00                  # Maximale Laufzeit (HH:MM:SS)
#SBATCH --output=test_td3_impl_%j.log    # Logdatei mit Job-ID

module purge

source ~/.bashrc

conda activate rl_env

# load api key from .env file
set -a
source .env
set +a

wandb login
# Run your training script
python ~/drl_project/main.py
