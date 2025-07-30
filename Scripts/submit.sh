#!/bin/bash
###############################################################################
# SLURM Job Submission Script (Using Cluster Default Python)
#
# This script splits a run range into multiple jobs and submits each job via 
# SLURM's sbatch command. Each job calls Read_Cut_Hist_D2O.py with a calculated 
# start and end run along with the analysis type (M1 or M2).
###############################################################################

###############################################################################
# Hardcoded Run Parameters
###############################################################################
start_run=19520
end_run=19529
M1_or_M2="M1"  # Analysis type, can be 'M1' or 'M2'
njobs=1

# Calculate total runs and runs per job (using ceiling division).
total_runs=$(( end_run - start_run + 1 ))
runs_per_job=$(( (total_runs + njobs - 1) / njobs ))

echo "Total runs: $total_runs"
echo "Number of jobs: $njobs"
echo "Runs per job: $runs_per_job"

###############################################################################
# Function: submit_job
#
# Submits a single job to SLURM with the appropriate run range.
#
# Arguments:
#   $1 - Job number (for naming purposes)
#   $2 - Start run number for this job
#   $3 - End run number for this job
###############################################################################
submit_job() {
    local job_num=$1
    local job_start=$2
    local job_end=$3
    local M1_or_M2=$4
    
    echo "Submitting job ${job_num}: Runs ${job_start} to ${job_end}"
    
    # Submit the job using the cluster's default Python.
    sbatch -J "job_${job_num}" --wrap="python ../Codes/Read_Cut_Hist_D2O.py ${job_start} ${job_end} ${M1_or_M2}"
}

###############################################################################
# Main Loop: Submit Jobs
###############################################################################
job=0
current_run=$start_run

while [ $current_run -le $end_run ]; do
    job_start=$current_run
    job_end=$(( current_run + runs_per_job - 1 ))
    
    if [ $job_end -gt $end_run ]; then
        job_end=$end_run
    fi
    
    submit_job $job $job_start $job_end
    
    current_run=$(( job_end + 1 ))
    job=$(( job + 1 ))
done

echo "All jobs submitted."