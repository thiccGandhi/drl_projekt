#!/bin/bash
#SBATCH --job-name=fetch_sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --output=/home/ul/ul_student/ul_cep22/my_folders/drl_projekt/sweeps/%A_%a_slurm_out.txt
#SBATCH --error=/home/ul/ul_student/ul_cep22/my_folders/drl_projekt/sweeps/%A_%a_slurm_err.txt
# Usage (one command):
#   sbatch /home/ul/ul_student/ul_cep22/my_folders/drl_projekt/slurm_sweep_array.sh /path/to/sweeps/<STAMP or latest>
# The script auto-detects number of cfg_*.json and re-submits itself as an array.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "USAGE: $0 <CFG_DIR>   # directory containing cfg_000.json ... cfg_NNN.json" >&2
  exit 2
fi

BASE_DIR="/home/ul/ul_student/ul_cep22/my_folders/drl_projekt"
PYTHON_SCRIPT="${BASE_DIR}/main_pars.py"

CFG_DIR="$1"

# --- SUBMITTER MODE: if not an array task yet, compute size and submit array ---
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # Count configs
  NUM=$(ls -1 "${CFG_DIR}"/cfg_*.json 2>/dev/null | wc -l)
  if [[ "${NUM}" -eq 0 ]]; then
    echo "No cfg_*.json found in ${CFG_DIR}" >&2
    exit 1
  fi
  N=$((NUM - 1))
  echo "Submitting array 0-${N} for CFG_DIR=${CFG_DIR}"
  # Re-submit this script as an array; #SBATCH directives above will apply
  sbatch --array=0-"${N}" "$0" "${CFG_DIR}"
  echo "Submitted. Exiting submitter."
  exit 0
fi

# --- WORKER MODE: run one config indexed by SLURM_ARRAY_TASK_ID ---
IDX="${SLURM_ARRAY_TASK_ID}"

CFG_PATH="${CFG_DIR}/cfg_$(printf "%03d" ${IDX}).json"
if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Config not found: ${CFG_PATH}" >&2
  exit 4
fi

# Activate venv
source ~/my_folders/drl_venv_py312/bin/activate

# Env for Python
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
export CFG_PATH="${CFG_PATH}"

echo "[$(date)] Node: ${HOSTNAME} | Task ${SLURM_ARRAY_TASK_ID} | CFG_PATH=${CFG_PATH}"
python "${PYTHON_SCRIPT}"
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} completed."
