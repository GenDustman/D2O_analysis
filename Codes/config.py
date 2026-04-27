#!/usr/bin/env python3
"""
Configuration Parameters for D2O Analysis

This file serves as the single source of truth for all analysis parameters.
Both the processing script (Read_Cut_Hist_D2O_multi.py) and the master
aggregator (aggregate_master.py) will import their settings from here.
"""

# --- Channel Definitions ---
# These are the indices inside ROOT branches: area[i], pulseH[i], peakPosition[i]
# They describe the detector layout only. Analysis-specific channel selections are defined below.
PMT_CHANNELS  = list(range(0, 12))     # PMT channels: 0-11
SIPM_CHANNELS = list(range(12, 22))    # All SiPM channels: 12-21

# --- Run & Directory Configuration ---
DATA_DIR_M1 = "/raid1/genli/Data_D2O/M1_data"
DATA_DIR_M2 = "/raid1/genli/Data_D2O/M2_data"
# M1 input suffix:
# - set to "auto" to try candidates in SUFFIX_M1_CANDIDATES (in order)
# - or set to a fixed suffix string
suffix_M1 = "auto"
SUFFIX_M1_CANDIDATES = ["_processed_v5.root", "_processed_v4.root"]
suffix_M2 = "_processed_H2O_v5.root"

# --- Global Analysis Cuts ---
TIME_INTERVAL_CUT_NS = 512  # Pile-up cut in ns
muon_life = 2197  # Muon lifetime in ns
DELTA_T_CUT = (2400, 32000)      # (min_ns, max_ns), min_ns, max_ns should be n*DELTA_T_BIN_WIDTH_NS
# DELTA_T_CUT = (8*muon_life, 100*muon_life)      # (min_ns, max_ns)
# DELTA_T_CUT = (960, 10560)      # (min_ns, max_ns)
PE_CUT = (0, 1000)             # (min_pe, max_pe)
TIME_STD_CUT = 2.5 * 16        # Max standard deviation of PMT hit times in an event (ns)
MULTIPLICITY_SPE = 2         # P.E. threshold to count a PMT as "hit"
MULTIPLICITY_CUT = 12          # Minimum number of hit PMTs for an event

# --- Time Quantization ---
TIME_TICK_NS = 16                 # DAQ time granularity
DELTA_T_BIN_WIDTH_NS = 160         # Δt bin width; choose k*TIME_TICK_NS (e.g., 64, 80, 96...)
DELTA_T_LEFT_EDGE_NS = 0

# --- Analysis Toggles And Selections ---
PERFORM_THIN_VETO_ANALYSIS = False
THIN_VETO_CHANNELS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # thin-veto panel selection
THIN_VETO_THRESHOLD = 30.0

PERFORM_BRN_ANALYSIS = True
BRN_DELTA_T_RANGE = (832, 4992)   # (ns) candidate timing window aligned to TIME_TICK_NS and BRN_DELTA_T_BIN_WIDTH_NS
BRN_DELTA_T_BIN_WIDTH_NS = 64   # Δt bin width for BRN plots, must be multiple of TIME_TICK_NS
BRN_SIPM_THRESHOLD_ADC = 30.0   # PulseH threshold for a SiPM channel to be "triggered"
BRN_SIPM_CHANNELS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # BRN channel selection
ENABLE_EVENT61_SYNTHETIC_BIT = True  # BRN-only synthetic Event61 preprocessing toggle
EVENT61_CHANNEL_INDEX = 22
EVENT61_ADC_RANGE = (20.0, 40.0)
EVENT61_THRESHOLD_ADC = EVENT61_ADC_RANGE[0]  # Legacy lower-edge alias; the trigger uses the full EVENT61_ADC_RANGE window.

# --- Per-Run / Analysis Plot Settings ---
RUN_PLOT_CONFIG = {
    'cut_total_pe_hist': {
        'bins': 100,
        'range': PE_CUT,
        'logscale': True,
    },
    'veto_efficiency': {
        'bins': 20,
        'range': (1000, 2000),
    },
    'sipm_area_hist': {
        'hist_bins': 100,
        'hist_range': (-50, 2000),
    },
}

LOW_LIGHT_FIT_RANGE = (-50, 400)  # (min_adc, max_adc) for multi-Gaussian SPE fits

LOW_LIGHT_PLOT_CONFIG = {
    'figure_size': (22, 16),
    'dpi': 300,
    'suptitle_fontsize': 20,
    'channel_title_fontsize': 15,
    'axis_label_fontsize': 13,
    'tick_labelsize': 12,
    'legend_fontsize': 11,
    'annotation_fontsize': 11,
}

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

HIGHLIGHT_FIT_CONFIG = {
    'bins': 120,
    'hist_range': (0, 120),
    'sum_bins': 160,
    'sum_hist_range': (0, 1200),
    'fit_window_half_width_pe': 12.0,
    'min_fit_points': 6,
}

EVENT61_FIT_CONFIG = {
    'enabled': True,
    'bins': 200,
    'hist_range': (0, 200),
    'fit_range': (20, 40),
    'signal_range': (20, 40),
    'min_fit_points': 6,
    'figure_size': (10, 6),
    'dpi': 300,
    'logscale': False,
}

# --- Master Aggregate Fit Settings ---
DO_TAU_FIT = True
TAU_FIT_WINDOW = (DELTA_T_CUT[0] + DELTA_T_BIN_WIDTH_NS, 10000)  # (start_ns, end_ns) for lifetime fit

HIGHLIGHT_EVOLUTION_FIT_CONFIG = {
    'model': 'linear',
    'exp_initial_t0': 0.0,
    'exp_initial_tau': 20.0,
    'exp_tau_min': 1e-6,
}

MICHEL_EVOLUTION_FIT_CONFIG = {
    'model': 'linear',
    'exp_initial_t0': 0.0,
    'exp_initial_tau': 20.0,
    'exp_tau_min': 1e-6,
}

# --- Thin Veto Analysis Plot Settings ---
THIN_VETO_HIST_CONFIG = {
    'height_bins': 100,
    'height_range': (0, 1000),
    'area_bins': 100,
    'area_range': (0, 10000),
}

# --- BRN Analysis Plot Settings ---
BRN_HIST_CONFIG = {
    'area_bins': 100,
    'area_range': (-50, 2000),
    'heatmap_cmap': 'viridis',
    'heatmap_logscale': True,
}

# --- Master Aggregate Plot Settings ---
MASTER_PLOT_CONFIG = {
    'main_delta_t': {
        'range': DELTA_T_CUT,
        'bin_width_ns': DELTA_T_BIN_WIDTH_NS,
        'fit_window': TAU_FIT_WINDOW,
        'logscale': True,
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'main_total_pe': {
        'range': PE_CUT,
        'bins': RUN_PLOT_CONFIG['cut_total_pe_hist']['bins'],
        'fit_range': (RUN_PLOT_CONFIG['veto_efficiency']['range'][0] * 0.5, RUN_PLOT_CONFIG['veto_efficiency']['range'][1] * 0.5),
        'logscale': True,
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'total_pe_comparison': {
        'range': PE_CUT,
        'bins': RUN_PLOT_CONFIG['cut_total_pe_hist']['bins'],
        'logscale': True,
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'veto_efficiency': {
        'range': RUN_PLOT_CONFIG['veto_efficiency']['range'],
        'bins': RUN_PLOT_CONFIG['veto_efficiency']['bins'],
        'fit_range': (RUN_PLOT_CONFIG['veto_efficiency']['range'][0] * 0.5, RUN_PLOT_CONFIG['veto_efficiency']['range'][1] * 0.5),
        'y_range': (0.995, 1.002),
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'veto_efficiency_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'beam_on_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'michel_peak_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'mu1_evolution': {
        'figure_size': (13, 7),
        'legend_ncol': 3,
        'dpi': 300,
    },
    'low_light_channels': {
        'fit_range': LOW_LIGHT_FIT_RANGE,
        'figure_size': (20, 15),
        'dpi': 300,
    },
    'highlight_pe_channels': {
        'hist_range': HIGHLIGHT_FIT_CONFIG['hist_range'],
        'figure_size': (20, 15),
        'dpi': 300,
    },
    'highlight_pe_sum': {
        'hist_range': HIGHLIGHT_FIT_CONFIG['sum_hist_range'],
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'highlight_peak_evolution': {
        'figure_size': (13, 7),
        'legend_ncol': 3,
        'dpi': 300,
    },
    'event61_hist': {
        'hist_range': EVENT61_FIT_CONFIG['hist_range'],
        'fit_range': EVENT61_FIT_CONFIG['fit_range'],
        'signal_range': EVENT61_FIT_CONFIG['signal_range'],
        'figure_size': EVENT61_FIT_CONFIG['figure_size'],
        'dpi': EVENT61_FIT_CONFIG['dpi'],
        'logscale': True,
    },
    'event61_mean_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'event61_sigma_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'event61_signal_evolution': {
        'figure_size': (12, 6),
        'dpi': 300,
    },
    'sipm_area_hist': {
        'hist_bins': RUN_PLOT_CONFIG['sipm_area_hist']['hist_bins'],
        'hist_range': RUN_PLOT_CONFIG['sipm_area_hist']['hist_range'],
        'logscale': True,
        'figure_size': (20, 15),
        'dpi': 300,
    },
    'thin_veto_height_comparison': {
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'thin_veto_area_comparison': {
        'figure_size': (10, 6),
        'dpi': 300,
    },
    'brn_delta_t': {
        'channels': BRN_SIPM_CHANNELS,
        'range': BRN_DELTA_T_RANGE,
        'bin_width_ns': BRN_DELTA_T_BIN_WIDTH_NS,
        'figure_size': (18, 12),
        'dpi': 300,
    },
    'brn_area': {
        'channels': BRN_SIPM_CHANNELS,
        'range': BRN_HIST_CONFIG['area_range'],
        'bins': BRN_HIST_CONFIG['area_bins'],
        'figure_size': (18, 12),
        'dpi': 300,
    },
    'brn_delta_t_area': {
        'channels': BRN_SIPM_CHANNELS,
        'delta_t_range': BRN_DELTA_T_RANGE,
        'area_range': BRN_HIST_CONFIG['area_range'],
        'figure_size': (18, 12),
        'dpi': 300,
        'cmap': BRN_HIST_CONFIG['heatmap_cmap'],
        'logscale': BRN_HIST_CONFIG['heatmap_logscale'],
    },
}

# --- Backward-Compatible Aliases ---
BINS = RUN_PLOT_CONFIG['cut_total_pe_hist']['bins']
VETO_BINS = RUN_PLOT_CONFIG['veto_efficiency']['bins']
VETO_RANGE = RUN_PLOT_CONFIG['veto_efficiency']['range']
LOGSCALE_GENERAL = RUN_PLOT_CONFIG['cut_total_pe_hist']['logscale']
LOGSCALE_DT_AGG = MASTER_PLOT_CONFIG['main_delta_t']['logscale']
LOGSCALE_PE_AGG = MASTER_PLOT_CONFIG['main_total_pe']['logscale']
SIPM_HIST_CONFIG = RUN_PLOT_CONFIG['sipm_area_hist']