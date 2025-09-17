#!/bin/bash
###############################################################################
# Manual Job Submission Script for Legacy Runs (4500-4505)
# This version runs the analysis directly without SLURM
###############################################################################

# --- User-Defined Path Configuration ---
SCRIPT_DIR="/home/genli/D2O_analysis/Codes"

# Run Parameters for Legacy Analysis
start_run=5290
end_run=5295
M1_or_M2="M1"

# Data Directories
DATA_BASE_DIR="/raid1/genli/Data_D2O/M1_data"
if [ "$M1_or_M2" == "M2" ]; then
    DATA_BASE_DIR="/raid1/genli/Data_D2O/M2_data"
fi

# Create a Unique Top-Level Directory for this entire analysis
TOP_OUTPUT_DIR="${DATA_BASE_DIR}/analysis_legacy_${start_run}-${end_run}_${M1_or_M2}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TOP_OUTPUT_DIR"
echo "Legacy analysis batch output will be in: ${TOP_OUTPUT_DIR}"

# Run the processing script directly
echo "Running legacy processing for runs ${start_run} to ${end_run}..."
python ${SCRIPT_DIR}/Read_Cut_Hist_D2O_multi_veto_legacy.py ${start_run} ${end_run} ${M1_or_M2} ${TOP_OUTPUT_DIR}

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "Processing completed successfully. Running aggregation..."
    python ${SCRIPT_DIR}/aggregate_master_veto_legacy.py ${TOP_OUTPUT_DIR}
    
    if [ $? -eq 0 ]; then
        echo "Aggregation completed successfully!"
        echo "Results are available in: ${TOP_OUTPUT_DIR}"
    else
        echo "Aggregation failed!"
    fi
else
    echo "Processing failed!"
fi

echo ""
echo "Note: This script uses legacy versions of the analysis code for older ROOT file format (v3)."
echo "Final output directory: ${TOP_OUTPUT_DIR}"
