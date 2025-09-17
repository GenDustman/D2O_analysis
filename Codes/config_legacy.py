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

# --- Cut & Binning Configuration ---
DELTA_T_CUT = (0, 10000)      # (min_ns, max_ns)
PE_CUT = (0, 2000)             # (min_pe, max_pe)
TIME_STD_CUT = 2.5 * 16        # Max standard deviation of PMT hit times in an event (ns)
MULTIPLICITY_SPE = 1.0         # P.E. threshold to count a PMT as "hit"
MULTIPLICITY_CUT = 11          # Minimum number of hit PMTs for an event, higher because chanel 5 is problematic, mu_1 fit is very small

# --- Histogram & Plotting Configuration ---
BINS = 100                     # General purpose bin count for histograms
VETO_BINS = 20                 # Bin count for veto efficiency plots
VETO_RANGE = (800, 2000)       # P.E. range for plotting veto efficiency
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
PERFORM_THIN_VETO_ANALYSIS = True
THIN_VETO_CHANNELS = [13, 14] # List of channels to analyze
THIN_VETO_THRESHOLD = 30.0       # Threshold for the veto panels in the list
# ADJ_CH_THRESHOLD is no longer used.

THIN_VETO_HIST_CONFIG = {
    'height_bins': 100,
    'height_range': (0, 1000),
    'area_bins': 100,
    'area_range': (0, 10000),
}