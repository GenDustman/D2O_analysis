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

# --- Channel Definitions ---
# These are the indices inside ROOT branches: area[i], pulseH[i], peakPosition[i]

PMT_CHANNELS  = list(range(0, 12))     # PMT channels: 0-11
SIPM_CHANNELS = list(range(12, 22))    # SiPM channels: 12-21

# --- Run & Directory Configuration ---
DATA_DIR_M1 = "/raid1/genli/Data_D2O/M1_data"
DATA_DIR_M2 = "/raid1/genli/Data_D2O/M2_data"
# M1 input suffix:
# - set to "auto" to try candidates in SUFFIX_M1_CANDIDATES (in order)
# - or set to a fixed suffix string
suffix_M1 = "auto"
SUFFIX_M1_CANDIDATES = ["_processed_v5.root", "_processed_v4.root"]
suffix_M2 = "_processed_H2O_v5.root"
# --- Cut & Binning Configuration ---
TIME_INTERVAL_CUT_NS = 2000  # Pile-up cut in ns
muon_life = 2197  # Muon lifetime in ns
DELTA_T_CUT = (2400, 32000)      # (min_ns, max_ns), min_ns, max_ns should be n*DELTA_T_BIN_WIDTH_NS
# DELTA_T_CUT = (8*muon_life, 100*muon_life)      # (min_ns, max_ns)
# DELTA_T_CUT = (960, 10560)      # (min_ns, max_ns)
PE_CUT = (0, 2000)             # (min_pe, max_pe)
TIME_STD_CUT = 2.5 * 16        # Max standard deviation of PMT hit times in an event (ns)
MULTIPLICITY_SPE = 2         # P.E. threshold to count a PMT as "hit"
MULTIPLICITY_CUT = 12          # Minimum number of hit PMTs for an event

# --- Time quantization & dedicated Δt binning ---
TIME_TICK_NS = 16                 # DAQ time granularity
DELTA_T_BIN_WIDTH_NS = 160         # Δt bin width; choose k*TIME_TICK_NS (e.g., 64, 80, 96...)
# Optional: if you ever want an offset (usually 0), keep it a multiple of TIME_TICK_NS
DELTA_T_LEFT_EDGE_NS = 0

# --- Histogram & Plotting Configuration ---
BINS = 100                     # or 'auto' or keep an int like 100
VETO_BINS = 20                 # Bin count for veto efficiency plots
VETO_RANGE = (1000, 2000)       # P.E. range for plotting veto efficiency
LOGSCALE_PE_AGG = True        # Use log scale for the aggregated P.E. y-axis
LOGSCALE_DT_AGG = True         # Use log scale for the aggregated delta_t y-axis
LOGSCALE_GENERAL = True        # Default log scale for per-run histograms

# --- Fitting Configuration ---
DO_TAU_FIT = True
TAU_FIT_WINDOW = (DELTA_T_CUT[0] + DELTA_T_BIN_WIDTH_NS, 10000)  # (start_ns, end_ns) for the lifetime fit
LOW_LIGHT_FIT_RANGE = (-50, 400) # (min_adc, max_adc) for multi-Gaussian SPE fits

# --- SiPM Analysis Configuration ---
SIPM_HIST_CONFIG = {
    'hist_bins': 100,
    'hist_range': (-50, 4000) # (min_adc, max_adc)
}

# --- SiPM pulseH Landau fit (triggerBits >= 32) ---
SIPM_PULSEH_FIT_CONFIG = {
    'enabled': True,
    'bins': 200,
    'hist_range': (0, 800),
    'threshold': 25.0,
    'fit_ranges_by_panel': {
        'top': (120, 800),
        'wide': (160, 800),
        'thin': (90, 800),
    },
    'mpv_bounds_by_panel': {
        'top': (120, 300),
        'wide': (160, 400),
        'thin': (90, 300),
    }
}

# --- Highlight (triggerBits == 8) PMT P.E. analysis ---
HIGHLIGHT_FIT_CONFIG = {
    'bins': 120,
    'hist_range': (0, 120),
    'fit_window_half_width_pe': 12.0,
    'min_fit_points': 6,
}

# --- Highlight peak evolution fit (master aggregation) ---
# model: "linear" or "exp"
# exp model: a * exp((x + t0) / tau) + b
HIGHLIGHT_EVOLUTION_FIT_CONFIG = {
    'model': 'linear',
    'exp_initial_t0': 0.0,
    'exp_initial_tau': 20.0,
    'exp_tau_min': 1e-6,
}

# --- Michel peak evolution fit (master aggregation) ---
# model: "linear" or "exp"
# exp model: a * exp(-(x + t0) / tau) + b
MICHEL_EVOLUTION_FIT_CONFIG = {
    'model': 'linear',
    'exp_initial_t0': 0.0,
    'exp_initial_tau': 20.0,
    'exp_tau_min': 1e-6,
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
BRN_SIPM_CHANNELS = SIPM_CHANNELS
BRN_HIST_CONFIG = {
    'area_bins': 100,
    'area_range': (-50, 4000) # (min_adc, max_adc)
}