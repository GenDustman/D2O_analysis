#!/bin/bash
set -euo pipefail
###############################################################################
# SLURM Job Submission Script for Master Aggregation ONLY
#
# Use this script to re-run the final aggregation step on an existing
# analysis folder (one that already contains all the 'subjob_*' outputs).
###############################################################################

# --- User-Defined Configuration ---

# 1. Set the absolute path to the directory containing your python scripts
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"
PYTHON_BIN="/raid1/genli/conda/miniconda3/envs/py311/bin/python"

# 2. Set the absolute path to the top-level analysis folder
#    (This is the folder that contains all the 'subjob_*' directories)
ANALYSIS_DIR="/raid1/genli/Data_D2O/M2_data/analysis_5616-6359_step1_M2_20260407-020113"

# 3. (Optional) Set a Job Name and Memory Request
#    Given the previous MemoryError, requesting more memory is a good idea.
#    Adjust "32G" as needed for your system (e.g., "16G", "64G").
JOB_NAME="master_agg"
MEMORY_REQ="32G"
PARTITION="blue"

# --- End of Configuration ---

if [[ $# -ge 1 ]]; then
       ANALYSIS_DIR="$1"
fi

if [[ $# -ge 2 ]]; then
       JOB_NAME="$2"
fi

if [[ ! -d "${ANALYSIS_DIR}" ]]; then
       echo "Error: analysis directory does not exist: ${ANALYSIS_DIR}" >&2
       exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/aggregate_master_veto.py" ]]; then
       echo "Error: aggregation script not found in ${SCRIPT_DIR}" >&2
       exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
       echo "Error: Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
       exit 1
fi

if ! find "${ANALYSIS_DIR}" -maxdepth 1 -type d -name 'subjob_*' | grep -q .; then
       echo "Error: no subjob_* directories found in ${ANALYSIS_DIR}" >&2
       exit 1
fi

analysis_base_name="$(basename "${ANALYSIS_DIR}")"
if [[ "${JOB_NAME}" == "master_agg" ]]; then
       JOB_NAME="master_${analysis_base_name}"
fi

echo "Submitting master aggregation job for directory:"
echo "${ANALYSIS_DIR}"
echo "Using scripts from:"
echo "${SCRIPT_DIR}"
echo "Using Python interpreter:"
echo "${PYTHON_BIN}"

sbatch -p "${PARTITION}" \
          -J "${JOB_NAME}" \
          --mem="${MEMORY_REQ}" \
          --wrap="cd ${SCRIPT_DIR} && ${PYTHON_BIN} ${SCRIPT_DIR}/aggregate_master_veto.py ${ANALYSIS_DIR}"

echo "Aggregation job has been submitted."