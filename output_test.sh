#!/bin/bash
#SBATCH --job-name=output_test            # Name des Jobs
#SBATCH --nodes=1                        # Anzahl der Nodes
#SBATCH --ntasks=1                       # Anzahl der Tasks
#SBATCH --cpus-per-task=4                # CPU-Kerne pro Task
#SBATCH --mem=2G                        # Arbeitsspeicher pro Task
#SBATCH --partition=cpu                  # Partition auf dem das ganze läuft
#SBATCH --time=48:00:00                  # Maximale Laufzeit (HH:MM:SS)
#SBATCH --output=output_test%j.log    # Logdatei mit Job-ID

module purge

source ~/.bashrc

conda activate rl_env


# Run your training script
python ~/drl_project/output_test.py
