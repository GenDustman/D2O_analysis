#!/bin/bash
###############################################################################
# Local Analysis Script (No SLURM Required)
#
# Runs the D2O analysis pipeline directly on the local machine:
#   1. Process ROOT files with Read_Cut_Hist_D2O_multi_veto.py
#   2. Aggregate results with aggregate_master_veto.py
#
# Usage: bash Scripts/run_local.sh
#        (edit the parameters below before running)
###############################################################################

set -e

# --- User Configuration ---
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

start_run=31235
end_run=31235
step=1
M1_or_M2="M1"
njobs=1   # Number of sequential chunks (for memory management; 1 = single process)

PYTHON_BIN="python3"

# --- Setup output directory ---
if [ "$M1_or_M2" == "M1" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M1_data"
elif [ "$M1_or_M2" == "M2" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M2_data"
else
    echo "ERROR: M1_or_M2 must be 'M1' or 'M2'." >&2
    exit 1
fi

TOP_OUTPUT_DIR="${DATA_BASE_DIR}/analysis_${start_run}-${end_run}_step${step}_${M1_or_M2}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TOP_OUTPUT_DIR"
echo "Analysis output will be in: ${TOP_OUTPUT_DIR}"

# --- Build run list ---
run_list=()
for ((r = start_run; r <= end_run; r += step)); do
    run_list+=($r)
done
total_runs=${#run_list[@]}
runs_per_job=$(( (total_runs + njobs - 1) / njobs ))

echo "Total runs: ${total_runs} (every ${step}th run from ${start_run} to ${end_run})"
echo "Number of sequential chunks: ${njobs} (${runs_per_job} runs each)"

# --- Stage B: Processing ---
echo ""
echo "=== Stage B: Processing ROOT files ==="

job=0
idx=0
PROCESSING_FAILED=0
while [ $idx -lt $total_runs ]; do
    job_start_run=${run_list[$idx]}
    end_idx=$(( idx + runs_per_job - 1 ))
    if [ $end_idx -ge $total_runs ]; then
        end_idx=$(( total_runs - 1 ))
    fi
    job_end_run=${run_list[$end_idx]}

    echo ""
    echo "--- Chunk $((job + 1))/${njobs}: Runs ${job_start_run} to ${job_end_run} ---"
    ${PYTHON_BIN} ${SCRIPT_DIR}/Read_Cut_Hist_D2O_multi_veto.py \
        ${job_start_run} ${job_end_run} ${M1_or_M2} ${TOP_OUTPUT_DIR} ${step} \
        || { echo "ERROR: Processing chunk $((job + 1)) failed." >&2; PROCESSING_FAILED=1; break; }

    idx=$(( end_idx + 1 ))
    job=$(( job + 1 ))
done

if [ $PROCESSING_FAILED -ne 0 ]; then
    echo "Processing did not complete successfully." >&2
    exit 1
fi

echo ""
echo "All ${njobs} processing chunk(s) completed."

# --- Stage C: Aggregation ---
echo ""
echo "=== Stage C: Aggregating results ==="
${PYTHON_BIN} ${SCRIPT_DIR}/aggregate_master_veto.py ${TOP_OUTPUT_DIR} \
    || { echo "ERROR: Aggregation failed." >&2; exit 1; }

echo ""
echo "=== Analysis complete ==="
echo "Results: ${TOP_OUTPUT_DIR}"
