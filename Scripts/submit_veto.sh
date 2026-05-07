#!/bin/bash
###############################################################################
# SLURM Job Submission Script with Automated Aggregation
#
# This version supports non-consecutive runs with a step parameter.
###############################################################################

# --- User-Defined Path Configuration ---
# NEW: Define the absolute path to the directory containing your python scripts.
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"
RUN_SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

# Hardcoded Run Parameters
start_run=7100
end_run=7104
step=1  # NEW: Process every Nth run
M1_or_M2="M2"
njobs=10
partition="blue"

# Data Directories
DATA_BASE_DIR="/raid1/genli/Data_D2O/M1_data"
if [ "$M1_or_M2" == "M2" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M2_data"
fi

# Create a Unique Top-Level Directory for this entire analysis
TOP_OUTPUT_DIR="${DATA_BASE_DIR}/analysis_${start_run}-${end_run}_step${step}_${M1_or_M2}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TOP_OUTPUT_DIR"
echo "Analysis batch output will be in: ${TOP_OUTPUT_DIR}"

# --- Code Snapshot ---
# Copy code to the output directory to isolate this run from future code changes
SNAPSHOT_DIR="${TOP_OUTPUT_DIR}/code"
mkdir -p "${SNAPSHOT_DIR}"
cp "${SCRIPT_DIR}/"*.py "${SNAPSHOT_DIR}/"
echo "Code snapshot created in: ${SNAPSHOT_DIR}"
echo "Jobs will run using code from: ${RUN_SCRIPT_DIR}"

# Calculate total runs with step
run_list=()
for ((r=start_run; r<=end_run; r+=step)); do
    run_list+=($r)
done
total_runs=${#run_list[@]}
runs_per_job=$(( (total_runs + njobs - 1) / njobs ))
echo "Total runs to process: $total_runs (every ${step}th run from $start_run to $end_run)"
echo "Runs per job: $runs_per_job"

# Array to hold job IDs
declare -a JOB_IDS=()

# Main Loop: Submit Parallel Processing Jobs
job=0
idx=0
while [ $idx -lt $total_runs ]; do
    job_start_run=${run_list[$idx]}
    end_idx=$(( idx + runs_per_job - 1 ))
    if [ $end_idx -ge $total_runs ]; then
        end_idx=$(( total_runs - 1 ))
    fi
    job_end_run=${run_list[$end_idx]}

    echo "Submitting processing job ${job}: Runs ${job_start_run} to ${job_end_run} with step ${step}"
    
    # Pass step parameter to Python script
    JOB_ID=$(sbatch -p "$partition" --parsable -J "job_${job}_${M1_or_M2}" --wrap="python3 ${RUN_SCRIPT_DIR}/Read_Cut_Hist_D2O_multi_veto.py ${job_start_run} ${job_end_run} ${M1_or_M2} ${TOP_OUTPUT_DIR} ${step}")
    
    JOB_IDS+=($JOB_ID)
    idx=$(( end_idx + 1 ))
    job=$(( job + 1 ))
done

echo "All ${#JOB_IDS[@]} processing jobs submitted."

# Convert job IDs to a colon-separated list
dependency_list=$(IFS=:; echo "${JOB_IDS[*]}")

# Submit the final aggregation job with a dependency
echo "Submitting final aggregation job with dependency list: ${dependency_list}"

sbatch -p "$partition" --dependency=afterok:${dependency_list} \
       -J "aggregate_${M1_or_M2}" \
    --wrap="python3 ${RUN_SCRIPT_DIR}/aggregate_master_veto.py ${TOP_OUTPUT_DIR}"

echo "Aggregation job has been submitted. It will run automatically after the others complete."