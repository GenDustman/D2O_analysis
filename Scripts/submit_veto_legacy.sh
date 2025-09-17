#!/bin/bash
###############################################################################
# SLURM Job Submission Script for Legacy Runs (4500-4505)
#
# This version uses the legacy scripts for older ROOT file format (v3)
###############################################################################

# --- User-Defined Path Configuration ---
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

# Hardcoded Run Parameters for Legacy Analysis
start_run=4123
end_run=4128
M1_or_M2="M1"
njobs=3

# Data Directories
DATA_BASE_DIR="/raid1/genli/Data_D2O/M1_data"
if [ "$M1_or_M2" == "M2" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M2_data"
fi

# Create a Unique Top-Level Directory for this entire analysis
TOP_OUTPUT_DIR="${DATA_BASE_DIR}/analysis_legacy_${start_run}-${end_run}_${M1_or_M2}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TOP_OUTPUT_DIR"
echo "Legacy analysis batch output will be in: ${TOP_OUTPUT_DIR}"

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

    echo "Submitting legacy processing job ${job}: Runs ${job_start} to ${job_end}"
    
    # Use the legacy script for older ROOT files
    JOB_ID=$(sbatch --parsable -J "legacy_job_${job}_${M1_or_M2}" --wrap="python ${SCRIPT_DIR}/Read_Cut_Hist_D2O_multi_veto_legacy.py ${job_start} ${job_end} ${M1_or_M2} ${TOP_OUTPUT_DIR}")
    
    JOB_IDS+=($JOB_ID)
    current_run=$(( job_end + 1 ))
    job=$(( job + 1 ))
done

echo "All ${#JOB_IDS[@]} legacy processing jobs submitted."

# Convert job IDs to a colon-separated list
dependency_list=$(IFS=:; echo "${JOB_IDS[*]}")

# Submit the final aggregation job with a dependency
echo "Submitting legacy aggregation job with dependency list: ${dependency_list}"

# Use the legacy aggregation script
sbatch --dependency=afterok:${dependency_list} \
       -J "aggregate_legacy_${M1_or_M2}" \
       --wrap="python ${SCRIPT_DIR}/aggregate_master_veto_legacy.py ${TOP_OUTPUT_DIR}"

echo "Legacy aggregation job has been submitted. It will run automatically after the others complete."
echo ""
echo "Note: This script uses legacy versions of the analysis code for older ROOT file format (v3)."
echo "Output will be in: ${TOP_OUTPUT_DIR}"
