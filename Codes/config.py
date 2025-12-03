#!/usr/bin/env python3
"""
Configuration Parameters for D2O Analysis

This file serves as the single source of truth for all analysis parameters.
Both the processing script (Read_Cut_Hist_D2O_multi.py) and the master
aggregator (aggregate_master.py) will import their settings from here.
"""

# --- Run & Directory Configuration ---
# These are used by submit.sh, but are here for reference.
# START_RUN = 19520
# END_RUN = 19820
# M1_or_M2 = "M1"
# N_JOBS = 30

# --- Run & Directory Configuration ---
DATA_DIR_M1 = "/raid1/genli/Data_D2O/M1_data"
DATA_DIR_M2 = "/raid1/genli/Data_D2O/M2_data"
suffix_M1 = "_processed_v4.root"
suffix_M2 = "_processed_H2O_v5.root"
# --- Cut & Binning Configuration ---
TIME_INTERVAL_CUT_NS = 2000  # Pile-up cut in ns
muon_life = 2197  # Muon lifetime in ns
DELTA_T_CUT = (muon_life, 5*muon_life)      # (min_ns, max_ns)
# DELTA_T_CUT = (8*muon_life, 100*muon_life)      # (min_ns, max_ns)
# DELTA_T_CUT = (960, 10560)      # (min_ns, max_ns)
PE_CUT = (0, 1000)             # (min_pe, max_pe)
TIME_STD_CUT = 2.5 * 16        # Max standard deviation of PMT hit times in an event (ns)
MULTIPLICITY_SPE = 1.5         # P.E. threshold to count a PMT as "hit"
MULTIPLICITY_CUT = 11          # Minimum number of hit PMTs for an event

# --- Time quantization & dedicated Δt binning ---
TIME_TICK_NS = 16                 # DAQ time granularity
DELTA_T_BIN_WIDTH_NS = 160         # Δt bin width; choose k*TIME_TICK_NS (e.g., 64, 80, 96...)
# Optional: if you ever want an offset (usually 0), keep it a multiple of TIME_TICK_NS
DELTA_T_LEFT_EDGE_NS = 0

# --- Histogram & Plotting Configuration ---
BINS = 1000                     # or 'auto' or keep an int like 100
VETO_BINS = 20                 # Bin count for veto efficiency plots
VETO_RANGE = (900, 2000)       # P.E. range for plotting veto efficiency
LOGSCALE_PE_AGG = False        # Use log scale for the aggregated P.E. y-axis
LOGSCALE_DT_AGG = True         # Use log scale for the aggregated delta_t y-axis
LOGSCALE_GENERAL = True        # Default log scale for per-run histograms

# --- Fitting Configuration ---
DO_TAU_FIT = True
TAU_FIT_WINDOW = (2500, 10000)  # (start_ns, end_ns) for the lifetime fit
LOW_LIGHT_FIT_RANGE = (-50, 400) # (min_adc, max_adc) for multi-Gaussian SPE fits

# --- SiPM Analysis Configuration ---
SIPM_HIST_CONFIG = {
    'hist_bins': 100,
    'hist_range': (-50, 4000) # (min_adc, max_adc)
}
# --- Veto Performance Analysis ---
PERFORM_THIN_VETO_ANALYSIS = False
THIN_VETO_CHANNELS = [12, 13, 14, 15]   # List of channels to analyze
THIN_VETO_THRESHOLD = 30.0       # Threshold for the veto panels in the list

THIN_VETO_HIST_CONFIG = {
    'height_bins': 100,
    'height_range': (0, 1000),
    'area_bins': 100,
    'area_range': (0, 10000),
}

# --- BRN (Beam-Related Neutron) Analysis ---
PERFORM_BRN_ANALYSIS = False
BRN_DELTA_T_RANGE = (0, 4000)   # (ns) Time window to plot BRN delta_t
BRN_DELTA_T_BIN_WIDTH_NS = 64    # Δt bin width for BRN analysis, must be multiple of TIME_TICK_NS
BRN_SIPM_THRESHOLD_ADC = 30.0     # PulseH threshold for a SiPM channel to be "triggered"
BRN_SIPM_CHANNELS = list(range(12, 22)) # Channels 12-21
BRN_HIST_CONFIG = {
    'area_bins': 100,
    'area_range': (-50, 4000) # (min_adc, max_adc)
}