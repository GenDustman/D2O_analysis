#!/bin/bash
###############################################################################
# SLURM Job Submission Script with Automated Aggregation
#
# This version defines key paths as variables for easier modification.
###############################################################################

# --- User-Defined Path Configuration ---
# NEW: Define the absolute path to the directory containing your python scripts.
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

# Hardcoded Run Parameters
start_run=1544
end_run=1548
M1_or_M2="M2"
njobs=3

# Data Directories
DATA_BASE_DIR="/raid1/genli/Data_D2O/M1_data"
if [ "$M1_or_M2" == "M2" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M2_data"
fi

# Create a Unique Top-Level Directory for this entire analysis
TOP_OUTPUT_DIR="${DATA_BASE_DIR}/analysis_${start_run}-${end_run}_${M1_or_M2}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TOP_OUTPUT_DIR"
echo "Analysis batch output will be in: ${TOP_OUTPUT_DIR}"

# Calculate runs per job
total_runs=$(( end_run - start_run + 1 ))
runs_per_job=$(( (total_runs + njobs - 1) / njobs ))
echo "Total runs: $total_runs, Runs per job: $runs_per_job"

# Array to hold job IDs
declare -a JOB_IDS=()

# Main Loop: Submit Parallel Processing Jobs
job=0
current_run=$start_run
while [ $current_run -le $end_run ]; do
    job_start=$current_run
    job_end=$(( current_run + runs_per_job - 1 ))
    if [ $job_end -gt $end_run ]; then
        job_end=$end_run
    fi

    echo "Submitting processing job ${job}: Runs ${job_start} to ${job_end}"
    
    # MODIFIED: Use the SCRIPT_DIR variable
    JOB_ID=$(sbatch --parsable -J "job_${job}_${M1_or_M2}" --wrap="python ${SCRIPT_DIR}/Read_Cut_Hist_D2O_multi_veto.py ${job_start} ${job_end} ${M1_or_M2} ${TOP_OUTPUT_DIR}")
    
    JOB_IDS+=($JOB_ID)
    current_run=$(( job_end + 1 ))
    job=$(( job + 1 ))
done

echo "All ${#JOB_IDS[@]} processing jobs submitted."

# Convert job IDs to a colon-separated list
dependency_list=$(IFS=:; echo "${JOB_IDS[*]}")

# Submit the final aggregation job with a dependency
echo "Submitting final aggregation job with dependency list: ${dependency_list}"

# MODIFIED: Use the SCRIPT_DIR variable
sbatch --dependency=afterok:${dependency_list} \
       -J "aggregate_${M1_or_M2}" \
       --wrap="python ${SCRIPT_DIR}/aggregate_master_veto.py ${TOP_OUTPUT_DIR}"

echo "Aggregation job has been submitted. It will run automatically after the others complete."