#!/bin/bash
###############################################################################
# SLURM Job Submission Script for Master Aggregation ONLY
#
# Use this script to re-run the final aggregation step on an existing
# analysis folder (one that already contains all the 'subjob_*' outputs).
###############################################################################

# --- User-Defined Configuration ---

# 1. Set the absolute path to the directory containing your python scripts
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

# 2. Set the absolute path to the top-level analysis folder
#    (This is the folder that contains all the 'subjob_*' directories)
ANALYSIS_DIR="/raid1/genli/Data_D2O/M1_data/BRN/analysis_25306-26233_M1_20251030-013056"

# 3. (Optional) Set a Job Name and Memory Request
#    Given the previous MemoryError, requesting more memory is a good idea.
#    Adjust "32G" as needed for your system (e.g., "16G", "64G").
JOB_NAME="master_agg"
MEMORY_REQ="32G"

# --- End of Configuration ---

echo "Submitting master aggregation job for directory:"
echo "${ANALYSIS_DIR}"
echo "Using scripts from:"
echo "${SCRIPT_DIR}"

sbatch -J "${JOB_NAME}" \
       --mem="${MEMORY_REQ}" \
       --wrap="python ${SCRIPT_DIR}/aggregate_master_veto.py ${ANALYSIS_DIR}"

echo "Aggregation job has been submitted."