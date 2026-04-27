#!/usr/bin/env python3
"""
Refactored script for processing ROOT files with detailed configuration
and an additional per-event time-std cut. Modular functions handle I/O,
histogram plotting, Δt computation, and aggregated τ fitting.
Includes low-light (triggerbit=16) analysis with multi-Gaussian fitting
and new 3x3 correlation maps for key variables with correlation coefficients.

MODIFICATION:
- Changed analysis from total charge (ADC) to total photoelectrons (P.E.).
- Error bars on P.E. histograms are calculated using simple Poisson counting
  statistics (sqrt(N)), neglecting the uncertainty from the P.E. calculation itself.
- ADDED: SiPM analysis for events with triggerbit >= 32, plotting area
  histograms for channels 12-21.
- ADDED: Aggregated "Total Photoelectron Comparison" plot.
- REVISED: Veto Efficiency plot is now generated ONLY after all quality cuts
  (multiplicity, P.E. range, time-std) have been applied.
- REVISED: Thin veto panel (Ch 20, 21) analysis now combines data from both channels
  and compares muon events (with coincidence) to all triggered events (no coincidence)
  using normalized histograms.
"""
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import uproot
import awkward as ak

try:
    from scipy.stats import landau as landau_dist
    HAVE_TRUE_LANDAU = True
except Exception:
    from scipy.stats import moyal as landau_dist
    HAVE_TRUE_LANDAU = False

# Import configuration or set defaults
try:
    import config
except ImportError:
    print("Warning: config module not found. Using default values.")
    class DefaultConfig:
        DATA_DIR_M1 = "/path/to/M1/data"
        DATA_DIR_M2 = "/path/to/M2/data"
        suffix_M1 = "auto"
        SUFFIX_M1_CANDIDATES = ["_processed_v5.root", "_processed_v4.root"]
        suffix_M2 = "_processed_H2O_v5.root"
        TIME_TICK_NS = 16
        PMT_CHANNELS  = list(range(0, 12))
        SIPM_CHANNELS = list(range(12, 22))
        DELTA_T_BIN_WIDTH_NS = 16
        DELTA_T_LEFT_EDGE_NS = 0
        DELTA_T_CUT = (0, 10000)
        PE_CUT = (0, 1000)
        BINS = 100
        VETO_BINS = 50
        VETO_RANGE = (0, 500)
        MULTIPLICITY_CUT = 1
        TIME_STD_CUT = 50
        TAU_FIT_WINDOW = (1000, 5000)
        LOGSCALE_DT_AGG = True
        LOGSCALE_PE_AGG = True
        DO_TAU_FIT = True
        LOW_LIGHT_FIT_RANGE = (-100, 1000)
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
        PERFORM_THIN_VETO_ANALYSIS = False
        PERFORM_BRN_ANALYSIS = False
        THIN_VETO_CHANNELS = [20, 21]
        THIN_VETO_THRESHOLD = 50
        THIN_VETO_HIST_CONFIG = {
            'height_range': (0, 1000),
            'height_bins': 100,
            'area_range': (0, 5000),
            'area_bins': 100
        }
        BRN_SIPM_CHANNELS = list(range(12, 22))
        BRN_SIPM_THRESHOLD_ADC = 50
        BRN_DELTA_T_RANGE = (0, 100000)
        BRN_DELTA_T_BIN_WIDTH_NS = 128
        ENABLE_EVENT61_SYNTHETIC_BIT = True
        EVENT61_CHANNEL_INDEX = 22
        EVENT61_ADC_RANGE = None
        EVENT61_THRESHOLD_ADC = 15.0
        BRN_HIST_CONFIG = {
            'area_range': (0, 5000),
            'area_bins': 100
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
    config = DefaultConfig()

PANEL_GROUP = {
    1: "top", 2: "top",
    3: "top", 4: "wide", 5: "top", 6: "top",
    7: "top", 8: "thin", 9: "thin", 10: "thin",
}


def _get_event61_adc_window():
    """Return the configured inclusive Event61 ADC window, with threshold fallback."""
    adc_range = getattr(config, 'EVENT61_ADC_RANGE', None)
    if isinstance(adc_range, (tuple, list)) and len(adc_range) == 2:
        try:
            adc_min = float(adc_range[0])
            adc_max = float(adc_range[1])
        except (TypeError, ValueError):
            adc_min = np.nan
            adc_max = np.nan
        if np.isfinite(adc_min) and np.isfinite(adc_max) and adc_max >= adc_min:
            return adc_min, adc_max

    adc_min = float(getattr(config, 'EVENT61_THRESHOLD_ADC', 15.0))
    return adc_min, np.inf


def _serialize_event61_adc_window(adc_min, adc_max):
    """Serialize the Event61 ADC window for JSON summaries."""
    return [float(adc_min), float(adc_max) if np.isfinite(adc_max) else None]


def _coerce_finite_window(window, fallback):
    """Return a finite two-sided window or the provided fallback."""
    if isinstance(window, (tuple, list)) and len(window) == 2:
        try:
            lo = float(window[0])
            hi = float(window[1])
        except (TypeError, ValueError):
            return fallback
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
    return fallback


def get_event61_fit_config(overrides=None):
    """Return normalized Event61 histogram/fit configuration."""
    cfg = dict(getattr(config, 'EVENT61_FIT_CONFIG', {}) or {})
    if isinstance(overrides, dict):
        cfg.update(overrides)

    hist_range = _coerce_finite_window(cfg.get('hist_range', (0, 200)), (0.0, 200.0))
    fit_range_default = _coerce_finite_window(getattr(config, 'EVENT61_ADC_RANGE', (20, 40)), (20.0, 40.0))
    fit_range = _coerce_finite_window(cfg.get('fit_range', fit_range_default), fit_range_default)
    signal_range = _coerce_finite_window(cfg.get('signal_range', fit_range), fit_range)

    return {
        'enabled': bool(cfg.get('enabled', True)),
        'bins': int(cfg.get('bins', 200)),
        'hist_range': hist_range,
        'fit_range': fit_range,
        'signal_range': signal_range,
        'min_fit_points': int(cfg.get('min_fit_points', 6)),
        'figure_size': tuple(cfg.get('figure_size', (10, 6))),
        'dpi': int(cfg.get('dpi', 300)),
        'logscale': bool(cfg.get('logscale', False)),
    }


def _event61_gaussian_plus_constant(x, amp, mu, sigma, c):
    sigma = np.maximum(sigma, 1e-6)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def _empty_event61_analysis(channel_index, channel_available, fit_config, n_entries=0):
    fit_lo, fit_hi = fit_config['fit_range']
    signal_lo, signal_hi = fit_config['signal_range']
    hist_lo, hist_hi = fit_config['hist_range']
    return {
        'channel_index': int(channel_index),
        'channel_available': bool(channel_available),
        'n_entries': int(n_entries),
        'fit_success': False,
        'fit_mean_adc': np.nan,
        'fit_mean_adc_err': np.nan,
        'fit_sigma_adc': np.nan,
        'fit_sigma_adc_err': np.nan,
        'fit_amplitude': np.nan,
        'fit_amplitude_err': np.nan,
        'fit_constant': np.nan,
        'fit_constant_err': np.nan,
        'fit_chi2': np.nan,
        'fit_ndof': np.nan,
        'fit_reduced_chi2': np.nan,
        'raw_window_count': 0.0,
        'background_window_count': np.nan,
        'background_window_count_err': np.nan,
        'background_subtracted_count': np.nan,
        'background_subtracted_count_err': np.nan,
        'fit_window': [float(fit_lo), float(fit_hi)],
        'signal_window': [float(signal_lo), float(signal_hi)],
        'hist_range': [float(hist_lo), float(hist_hi)],
        'bins': int(fit_config['bins']),
        'popt': None,
        'perr': None,
        'fit_error': None,
    }


def extract_event61_channel_values(pulseh_array, channel_index=None):
    """Extract finite Event61 pulse-height values from the configured channel."""
    pulseh = np.asarray(pulseh_array)
    event61_channel = int(getattr(config, 'EVENT61_CHANNEL_INDEX', 22) if channel_index is None else channel_index)
    if pulseh.ndim != 2 or event61_channel < 0 or event61_channel >= pulseh.shape[1]:
        return np.array([], dtype=float), event61_channel, False

    values = np.asarray(pulseh[:, event61_channel], dtype=float)
    values = values[np.isfinite(values)]
    return values, event61_channel, True


def build_event61_histogram_payload(pulseh_array, fit_config=None, channel_index=None):
    """Build a histogram-only Event61 payload from pulseH arrays."""
    cfg = get_event61_fit_config(fit_config)
    edges = np.linspace(cfg['hist_range'][0], cfg['hist_range'][1], cfg['bins'] + 1)
    counts = np.zeros(cfg['bins'], dtype=float)
    values, event61_channel, channel_available = extract_event61_channel_values(pulseh_array, channel_index=channel_index)

    if channel_available and values.size > 0:
        counts, _ = np.histogram(values, bins=edges)

    return {
        'counts': counts,
        'edges': edges,
        'channel_index': int(event61_channel),
        'channel_available': bool(channel_available),
        'n_entries': int(values.size),
    }


def analyze_event61_histogram_payload(hist_payload, fit_config=None):
    """Fit the Event61 histogram with a Gaussian plus constant background."""
    cfg = get_event61_fit_config(fit_config)
    counts = np.asarray(hist_payload.get('counts', []), dtype=float)
    edges = np.asarray(hist_payload.get('edges', []), dtype=float)
    channel_index = int(hist_payload.get('channel_index', getattr(config, 'EVENT61_CHANNEL_INDEX', 22)))
    channel_available = bool(hist_payload.get('channel_available', False))
    n_entries = int(hist_payload.get('n_entries', np.sum(counts)))
    summary = _empty_event61_analysis(channel_index, channel_available, cfg, n_entries=n_entries)

    if counts.size == 0 or edges.size != counts.size + 1:
        summary['fit_error'] = 'Invalid histogram payload'
        return summary

    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_lo, fit_hi = cfg['fit_range']
    signal_lo, signal_hi = cfg['signal_range']
    fit_mask = (centers >= fit_lo) & (centers <= fit_hi)
    signal_mask = (centers >= signal_lo) & (centers <= signal_hi)

    summary['raw_window_count'] = float(np.sum(counts[signal_mask]))
    if (not channel_available) or np.sum(counts) <= 0:
        summary['fit_error'] = 'No Event61 entries available'
        return summary
    if np.count_nonzero(fit_mask) < cfg['min_fit_points']:
        summary['fit_error'] = 'Too few bins in fit window'
        return summary
    if float(np.sum(counts[fit_mask])) <= 0.0:
        summary['fit_error'] = 'No counts in fit window'
        return summary

    x_fit = centers[fit_mask]
    y_fit = counts[fit_mask]
    amp_guess = max(float(np.max(y_fit) - np.min(y_fit)), 1.0)
    mu_guess = float(x_fit[np.argmax(y_fit)])
    sigma_guess = max(float(0.25 * (fit_hi - fit_lo)), 1.0)
    c_guess = max(float(np.min(y_fit)), 0.0)
    sigma_y = np.sqrt(np.clip(y_fit, 1.0, None))

    try:
        popt, pcov = curve_fit(
            _event61_gaussian_plus_constant,
            x_fit,
            y_fit,
            p0=[amp_guess, mu_guess, sigma_guess, c_guess],
            bounds=([0.0, fit_lo, 1e-3, 0.0], [np.inf, fit_hi, fit_hi - fit_lo, np.inf]),
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=30000,
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
        y_pred = _event61_gaussian_plus_constant(x_fit, *popt)
        chi2 = float(np.sum(((y_fit - y_pred) / sigma_y) ** 2))
        ndof = int(len(x_fit) - len(popt))
        red_chi2 = chi2 / ndof if ndof > 0 else np.nan
        background_sum = float(popt[3] * np.count_nonzero(signal_mask))
        background_sum_err = float(perr[3] * np.count_nonzero(signal_mask)) if len(perr) > 3 and np.isfinite(perr[3]) else np.nan
        background_subtracted = float(summary['raw_window_count'] - background_sum)
        if np.isfinite(background_sum_err):
            background_subtracted_err = float(np.sqrt(max(summary['raw_window_count'], 0.0) + background_sum_err ** 2))
        else:
            background_subtracted_err = float(np.sqrt(max(summary['raw_window_count'], 0.0)))

        summary.update({
            'fit_success': True,
            'fit_mean_adc': float(popt[1]),
            'fit_mean_adc_err': float(perr[1]) if len(perr) > 1 and np.isfinite(perr[1]) else np.nan,
            'fit_sigma_adc': float(popt[2]),
            'fit_sigma_adc_err': float(perr[2]) if len(perr) > 2 and np.isfinite(perr[2]) else np.nan,
            'fit_amplitude': float(popt[0]),
            'fit_amplitude_err': float(perr[0]) if len(perr) > 0 and np.isfinite(perr[0]) else np.nan,
            'fit_constant': float(popt[3]),
            'fit_constant_err': float(perr[3]) if len(perr) > 3 and np.isfinite(perr[3]) else np.nan,
            'fit_chi2': chi2,
            'fit_ndof': ndof,
            'fit_reduced_chi2': red_chi2,
            'background_window_count': background_sum,
            'background_window_count_err': background_sum_err,
            'background_subtracted_count': background_subtracted,
            'background_subtracted_count_err': background_subtracted_err,
            'popt': popt,
            'perr': perr,
        })
    except Exception as exc:
        summary['fit_error'] = str(exc)

    return summary


def plot_event61_histogram_payload(hist_payload, output_dir, file_label, M1_or_M2,
                                   fit_config=None, filename_suffix='event61_pulseh_fit',
                                   title_prefix='Event61 pulseH'):
    """Plot and save the Event61 histogram and fit summary."""
    cfg = get_event61_fit_config(fit_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = np.asarray(hist_payload.get('counts', []), dtype=float)
    edges = np.asarray(hist_payload.get('edges', []), dtype=float)
    analysis = analyze_event61_histogram_payload(hist_payload, cfg)
    filename_label = file_label.replace(' ', '_').replace('-', '_').replace(':', '')
    base_filename = f'{filename_label}_{M1_or_M2}_{filename_suffix}'

    fig, ax = plt.subplots(figsize=cfg['figure_size'])
    bin_width = float(np.median(np.diff(edges))) if edges.size > 1 else np.nan
    if counts.size > 0 and edges.size == counts.size + 1:
        ax.step(edges, np.append(counts, counts[-1] if len(counts) > 0 else 0.0), where='post', linewidth=1.6,
                color='navy', label=f'Finite entries: {analysis["n_entries"]:,}')

    fit_lo, fit_hi = cfg['fit_range']
    signal_lo, signal_hi = cfg['signal_range']
    ax.axvspan(signal_lo, signal_hi, color='gold', alpha=0.14, label=f'Signal window [{signal_lo:.0f}, {signal_hi:.0f}]')
    if (fit_lo, fit_hi) != (signal_lo, signal_hi):
        ax.axvspan(fit_lo, fit_hi, color='gray', alpha=0.12, label=f'Fit window [{fit_lo:.0f}, {fit_hi:.0f}]')

    if analysis.get('fit_success') and analysis.get('popt') is not None:
        x_plot = np.linspace(fit_lo, fit_hi, 500)
        ax.plot(
            x_plot,
            _event61_gaussian_plus_constant(x_plot, *analysis['popt']),
            color='crimson',
            linewidth=1.8,
            label=(
                f'Fit: $\\mu$={analysis["fit_mean_adc"]:.2f}±{analysis["fit_mean_adc_err"]:.2f}, '
                f'$\\sigma$={analysis["fit_sigma_adc"]:.2f}±{analysis["fit_sigma_adc_err"]:.2f}'
            )
        )

    textbox_lines = [
        f'Raw [{signal_lo:.0f}, {signal_hi:.0f}] count: {analysis["raw_window_count"]:.1f}',
        f'Background-subtracted: {analysis["background_subtracted_count"]:.1f} ± {analysis["background_subtracted_count_err"]:.1f}' if np.isfinite(analysis['background_subtracted_count']) else 'Background-subtracted: n/a',
        f'Constant background/bin: {analysis["fit_constant"]:.2f} ± {analysis["fit_constant_err"]:.2f}' if np.isfinite(analysis['fit_constant']) else 'Constant background/bin: n/a',
        f'Reduced $\\chi^2$: {analysis["fit_reduced_chi2"]:.2f}' if np.isfinite(analysis['fit_reduced_chi2']) else 'Reduced $\\chi^2$: n/a',
    ]
    ax.text(
        0.98,
        0.97,
        '\n'.join(textbox_lines),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75),
    )

    ax.set_title(f'{title_prefix} (Ch {analysis["channel_index"]}, {file_label}, {M1_or_M2})')
    ax.set_xlabel('pulseH (ADC)')
    if np.isfinite(bin_width):
        ax.set_ylabel(f'Events/{bin_width:g} ADC')
    else:
        ax.set_ylabel('Events')
    ax.set_xlim(cfg['hist_range'])
    if cfg['logscale']:
        ax.set_yscale('log')
        ax.set_ylim(0.8, None)
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.35)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.25)
    ax.minorticks_on()
    ax.legend(loc='upper left', fontsize='small')
    fig.tight_layout()

    img_save_path = output_dir / f'{base_filename}.png'
    pkl_save_path = output_dir / f'{base_filename}.pkl'
    summary_pkl_path = output_dir / f'{base_filename}_summary.pkl'
    summary_csv_path = output_dir / f'{base_filename}_summary.csv'

    fig.savefig(img_save_path, dpi=cfg['dpi'])
    plt.close(fig)

    with open(pkl_save_path, 'wb') as f:
        pickle.dump({'hist_payload': hist_payload, 'analysis': analysis}, f)

    summary_df = pd.DataFrame([{
        'channel_index': analysis['channel_index'],
        'channel_available': analysis['channel_available'],
        'n_entries': analysis['n_entries'],
        'fit_success': analysis['fit_success'],
        'fit_mean_adc': analysis['fit_mean_adc'],
        'fit_mean_adc_err': analysis['fit_mean_adc_err'],
        'fit_sigma_adc': analysis['fit_sigma_adc'],
        'fit_sigma_adc_err': analysis['fit_sigma_adc_err'],
        'fit_amplitude': analysis['fit_amplitude'],
        'fit_amplitude_err': analysis['fit_amplitude_err'],
        'fit_constant': analysis['fit_constant'],
        'fit_constant_err': analysis['fit_constant_err'],
        'fit_chi2': analysis['fit_chi2'],
        'fit_ndof': analysis['fit_ndof'],
        'fit_reduced_chi2': analysis['fit_reduced_chi2'],
        'raw_window_count': analysis['raw_window_count'],
        'background_window_count': analysis['background_window_count'],
        'background_window_count_err': analysis['background_window_count_err'],
        'background_subtracted_count': analysis['background_subtracted_count'],
        'background_subtracted_count_err': analysis['background_subtracted_count_err'],
        'fit_window_min': analysis['fit_window'][0],
        'fit_window_max': analysis['fit_window'][1],
        'signal_window_min': analysis['signal_window'][0],
        'signal_window_max': analysis['signal_window'][1],
        'hist_min': analysis['hist_range'][0],
        'hist_max': analysis['hist_range'][1],
        'bins': analysis['bins'],
        'fit_error': analysis['fit_error'],
    }])
    summary_df.to_pickle(summary_pkl_path)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f'Event61 pulseH fit saved to {img_save_path}')
    print(f'Event61 pulseH fit data saved to {pkl_save_path}')

    return analysis

class HistogramCalculator:
    """Handles histogram calculations and binning."""
    
    @staticmethod
    def calculate_histograms(data_dict, config_dict):
        """Calculates histograms based on configuration."""
        histograms = {}
        edges = {}
        for key, data in data_dict.items():
            if key.endswith('_h'):  # Height data
                bins = np.linspace(*config_dict['height_range'], config_dict['height_bins'] + 1)
                edges[key] = bins
            elif key.endswith('_a'):  # Area data
                bins = np.linspace(*config_dict['area_range'], config_dict['area_bins'] + 1)
                edges[key] = bins
            else:
                continue  # Skip if not height or area

            if data is not None and data.size > 0:
                histograms[key], _ = np.histogram(data, bins=bins)
            else:
                histograms[key] = np.zeros(len(bins) - 1)
                
        return histograms, edges

    @staticmethod
    def bin_edges_from_spec(bins_spec, data, data_range):
        """Return bin edges for np.histogram / plotting."""
        data = np.asarray(data)
        lo, hi = data_range
        if isinstance(bins_spec, (int, np.integer)):
            return np.linspace(lo, hi, int(bins_spec) + 1)
        if isinstance(bins_spec, str):
            return np.histogram_bin_edges(data[np.isfinite(data)], bins=bins_spec, range=(lo, hi))
        edges = np.asarray(bins_spec)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bins_spec must be int, rule string, or 1D edges array")
        return edges

    @staticmethod
    def make_dt_edges(delta_t_range):
        """Build Δt bin edges using a dedicated width aligned to the DAQ time tick."""
        dt_min, dt_max = delta_t_range
        tick = getattr(config, 'TIME_TICK_NS', 16)
        width = getattr(config, 'DELTA_T_BIN_WIDTH_NS', tick)
        left = getattr(config, 'DELTA_T_LEFT_EDGE_NS', 0)

        # Force width and left edge to the tick grid
        width = int(round(width / tick)) * tick
        width = max(width, tick)
        left = int(round(left / tick)) * tick

        # Snap the plotting window to the tick grid
        start = int(np.floor((dt_min - left) / tick)) * tick + left
        stop = int(np.ceil((dt_max - left) / tick)) * tick + left

        # Build edges (ensure we cover the full [dt_min, dt_max] range)
        edges = np.arange(start, stop + width, width)
        return edges

class FileHandler:
    """Handles file I/O operations."""
    
    @staticmethod
    def ensure_dir(path: Path):
        """Ensure that a directory exists; create it and any parent directories if necessary."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_pickle(data: dict, path: Path):
        """Serialize and save a Python dictionary to a pickle file."""
        with path.open('wb') as f:
            pickle.dump(data, f)

class DataProcessor:
    """Handles data processing operations."""
    
    def __init__(self):
        self.hist_calc = HistogramCalculator()
        self.file_handler = FileHandler()

    def calculate_total_pe(self, df, mu1_values):
        """Calculates the total photoelectrons for each event using per-channel gain."""
        if np.all(np.isnan(mu1_values)):
            print("ERROR: Low-light fit failed. Cannot calculate photoelectrons.")
            return np.full(len(df), np.nan)

        mu1_safe = np.where(np.isnan(mu1_values) | (mu1_values <= 0), np.inf, mu1_values)
        if np.any(mu1_safe == np.inf):
            nan_ch = np.where(np.isnan(mu1_values) | (mu1_values <= 0))[0]
            print(f"Warning: mu1 fit failed/invalid for channels {nan_ch}. These channels will be excluded from the P.E. sum.")
        
        area_data_np = np.array(df['area_array'].to_list())[:, config.PMT_CHANNELS]
        pe_per_channel = area_data_np / mu1_safe
        pe_per_channel = np.clip(pe_per_channel, 0.0, None)
        total_pe = np.sum(pe_per_channel, axis=1)
        
        return total_pe

    def compute_delta_t(self, df, muon_bits, veto_bits, mult_thresh):
        """Compute time differences Δt between veto events and the preceding muon event."""
        muon_mask = df['triggerBits'] >= muon_bits
        veto_mask = (df['triggerBits'] == veto_bits) & (df['multiplicity'] >= mult_thresh)
        muon_times = df.loc[muon_mask, 'nsTime'].values
        events = df.loc[veto_mask].copy()
        times = events['nsTime'].values
        idx = np.searchsorted(muon_times, times, side='right')
        delta_t = np.full(times.shape, np.nan)
        valid = idx > 0
        delta_t[valid] = times[valid] - muon_times[idx[valid] - 1]
        events['delta_t'] = delta_t
        return events

class ThinVetoAnalyzer:
    """Handles thin veto panel analysis."""
    
    @staticmethod
    def plot_thin_veto_performance(df, pulseh_array, area_array, output_dir, label, M1_or_M2,
                                  thin_veto_channels, threshold, mult_cut, hist_config):
        """
        Analyzes thin veto panel performance by comparing muon events with coincidence
        to all triggered events.
        """
        # Extract thin veto data
        thin_veto_mask = (df['multiplicity'] > mult_cut)
        thin_veto_events = df[thin_veto_mask]
        
        if thin_veto_events.empty:
            print(f"No events passed multiplicity cut for thin veto analysis in {label}.")
            return None
        
        thin_veto_indices = thin_veto_events.index
        thin_pulseh = pulseh_array[thin_veto_indices]
        thin_area = area_array[thin_veto_indices]
        
        # Combine data from all thin veto channels
        combined_pulseh = np.zeros(len(thin_veto_events))
        combined_area = np.zeros(len(thin_veto_events))
        
        for ch in thin_veto_channels:
            if ch < thin_pulseh.shape[1]:
                combined_pulseh += thin_pulseh[:, ch]
                combined_area += thin_area[:, ch]
        
        # Apply threshold
        above_threshold_mask = combined_pulseh > threshold
        
        # Muon events (with coincidence, triggerBits == 32)
        muon_mask = (thin_veto_events['triggerBits'] == 32) & above_threshold_mask
        muon_h = combined_pulseh[muon_mask]
        muon_a = combined_area[muon_mask]
        
        # All triggered events (no coincidence requirement)
        no_co_mask = above_threshold_mask
        no_co_h = combined_pulseh[no_co_mask]
        no_co_a = combined_area[no_co_mask]
        
        # Plot normalized comparison histograms
        FileHandler.ensure_dir(output_dir)
        filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
        
        # Height comparison
        height_bins = np.linspace(*hist_config['height_range'], hist_config['height_bins'] + 1)
        plt.figure(figsize=(10, 6))
        
        if muon_h.size > 0:
            plt.hist(muon_h, bins=height_bins, alpha=0.5, density=True, 
                    label=f'Muon Events (N={len(muon_h)})', color='blue')
        if no_co_h.size > 0:
            plt.hist(no_co_h, bins=height_bins, alpha=0.5, density=True,
                    label=f'All Triggered (N={len(no_co_h)})', color='red')
        
        plt.xlabel('Pulse Height (ADC)')
        plt.ylabel('Normalized Events')
        plt.title(f'Thin Veto Height Comparison - {label} ({M1_or_M2})')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename_label}_{M1_or_M2}_thin_veto_height.png')
        plt.close()
        
        # Area comparison
        area_bins = np.linspace(*hist_config['area_range'], hist_config['area_bins'] + 1)
        plt.figure(figsize=(10, 6))
        
        if muon_a.size > 0:
            plt.hist(muon_a, bins=area_bins, alpha=0.5, density=True,
                    label=f'Muon Events (N={len(muon_a)})', color='blue')
        if no_co_a.size > 0:
            plt.hist(no_co_a, bins=area_bins, alpha=0.5, density=True,
                    label=f'All Triggered (N={len(no_co_a)})', color='red')
        
        plt.xlabel('Pulse Area (ADC)')
        plt.ylabel('Normalized Events')
        plt.title(f'Thin Veto Area Comparison - {label} ({M1_or_M2})')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename_label}_{M1_or_M2}_thin_veto_area.png')
        plt.close()
        
        return muon_h, muon_a, no_co_h, no_co_a

class BRNAnalyzer:
    """Handles Beam-Related Neutron analysis."""

    @staticmethod
    def build_brn_trigger_info(trigger_bits, pulseh_array):
        """Build BRN-only trigger masks without mutating the shared triggerBits values."""
        trigger_bits_np = np.asarray(trigger_bits, dtype=np.int64)
        pulseh_array_np = np.asarray(pulseh_array)
        event_count = len(trigger_bits_np)
        event61_channel_index = int(getattr(config, 'EVENT61_CHANNEL_INDEX', 22))
        event61_adc_min, event61_adc_max = _get_event61_adc_window()
        event61_enabled = bool(getattr(config, 'ENABLE_EVENT61_SYNTHETIC_BIT', False))

        event61_mask = np.zeros(event_count, dtype=bool)
        channel_available = (
            pulseh_array_np.ndim == 2
            and pulseh_array_np.shape[0] == event_count
            and pulseh_array_np.shape[1] > event61_channel_index
        )
        if event61_enabled and channel_available:
            event61_values = np.asarray(pulseh_array_np[:, event61_channel_index], dtype=float)
            event61_mask = np.isfinite(event61_values) & (event61_values >= event61_adc_min)
            if np.isfinite(event61_adc_max):
                event61_mask &= event61_values <= event61_adc_max
        elif event61_enabled and event_count > 0:
            print(
                f"Event61 synthetic bit enabled, but pulseH[{event61_channel_index}] is unavailable. "
                "Using legacy BRN trigger logic."
            )

        if event61_enabled and channel_available:
            beam_on_mask = event61_mask
        else:
            beam_on_mask = trigger_bits_np == 1

        sipm_mask = np.isin(trigger_bits_np, [32, 34])

        return {
            'beam_on_mask': beam_on_mask,
            'sipm_mask': sipm_mask,
            'event61_mask': event61_mask,
            'event61_count': int(np.count_nonzero(event61_mask)),
            'event61_adc_min': event61_adc_min,
            'event61_adc_max': event61_adc_max,
            'event61_adc_range': _serialize_event61_adc_window(event61_adc_min, event61_adc_max),
            'event61_threshold_adc': event61_adc_min,
            'event61_channel_index': event61_channel_index,
            'event61_adjustment_applied': bool(event61_enabled and channel_available),
            'event61_channel_available': bool(channel_available),
        }
    
    @staticmethod
    def compute_brn_data(df, pulseh_array, area_array, channels_to_analyze, brn_threshold, brn_trigger_info=None):
        """
        Computes BRN delta_t and SiPM area data on a per-channel basis.
        - BRN delta_t: Time between SiPM event and previous BRN beam-on reference.
        - Data is stored only for channels that exceed the brn_threshold.
        """
        pulseh_array = np.asarray(pulseh_array)
        area_array = np.asarray(area_array)

        if len(df) != len(pulseh_array) or len(df) != len(area_array):
            raise ValueError(
                f"BRN input length mismatch: len(df)={len(df)}, "
                f"len(pulseh_array)={len(pulseh_array)}, len(area_array)={len(area_array)}"
            )

        if brn_trigger_info is None:
            brn_trigger_info = BRNAnalyzer.build_brn_trigger_info(df['triggerBits'].to_numpy(), pulseh_array)
        beam_on_mask = np.asarray(brn_trigger_info.get('beam_on_mask', np.zeros(len(df), dtype=bool)), dtype=bool)
        sipm_mask = np.asarray(brn_trigger_info.get('sipm_mask', np.zeros(len(df), dtype=bool)), dtype=bool)

        # Get times for BRN beam-on reference events and SiPM candidate events.
        beam_on_times = df.loc[beam_on_mask, 'nsTime'].values
        sipm_events = df.loc[sipm_mask].copy().reset_index(drop=True)

        if sipm_events.empty or beam_on_times.size == 0:
            print("No SiPM events or no beam-on events. Skipping BRN analysis.")
            return {}

        sipm_times = sipm_events['nsTime'].values
        
        # Compute BRN delta_t for all SiPM events
        idx = np.searchsorted(beam_on_times, sipm_times, side='right')
        delta_t = np.full(sipm_times.shape, np.nan)
        valid = idx > 0
        delta_t[valid] = sipm_times[valid] - beam_on_times[idx[valid] - 1]
        sipm_events['brn_delta_t'] = delta_t
        
        # Get the corresponding pulseH and area arrays for the SiPM events
        sipm_pulseh_array = pulseh_array[sipm_mask]
        sipm_area_array = area_array[sipm_mask]
        all_brn_delta_t = sipm_events['brn_delta_t'].values

        # Initialize data structure
        channel_data = {ch: {'delta_t': [], 'area': [], 'delta_t_area': []} for ch in channels_to_analyze}

        # Filter events by delta_t cut (apply to both delta_t and area data)
        dt_min, dt_max = config.BRN_DELTA_T_RANGE
        dt_cut_mask = ((sipm_events['brn_delta_t'] >= dt_min) & (sipm_events['brn_delta_t'] <= dt_max)).to_numpy(dtype=bool)
        events_in_dt_range = sipm_events.loc[dt_cut_mask]
        
        # Populate per-channel 1D and paired 2D BRN observables for events in the dt cut.
        if not events_in_dt_range.empty:
            filtered_pulseh_array_dt = sipm_pulseh_array[dt_cut_mask]
            filtered_area_array = sipm_area_array[dt_cut_mask]
            filtered_brn_delta_t = all_brn_delta_t[dt_cut_mask]

            for i in range(len(events_in_dt_range)):
                event_dt = filtered_brn_delta_t[i]
                if not np.isfinite(event_dt):
                    continue

                event_pulseh = filtered_pulseh_array_dt[i]
                event_area = filtered_area_array[i]
                for ch in channels_to_analyze:
                    if ch >= len(event_pulseh) or ch >= len(event_area) or event_pulseh[ch] <= brn_threshold:
                        continue
                    area_value = event_area[ch]
                    if not np.isfinite(area_value):
                        continue
                    channel_data[ch]['delta_t'].append(event_dt)
                    channel_data[ch]['area'].append(area_value)
                    channel_data[ch]['delta_t_area'].append((event_dt, area_value))

        # Convert lists to numpy arrays
        for ch in channels_to_analyze:
            channel_data[ch]['delta_t'] = np.array(channel_data[ch]['delta_t'])
            channel_data[ch]['area'] = np.array(channel_data[ch]['area'])
            channel_pairs = np.asarray(channel_data[ch]['delta_t_area'], dtype=float)
            if channel_pairs.size == 0:
                channel_pairs = np.empty((0, 2), dtype=float)
            else:
                channel_pairs = channel_pairs.reshape(-1, 2)
            channel_data[ch]['delta_t_area'] = channel_pairs

        return channel_data

    @staticmethod
    def plot_brn_histograms(channel_data, output_dir, label, M1_or_M2, brn_dt_range, hist_config):
        """Plots the per-channel BRN delta_t and area histograms."""
        if not channel_data:
            return
            
        FileHandler.ensure_dir(output_dir)
        filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
        channels_to_analyze = list(channel_data.keys())
        
        # Plot BRN Delta T Histograms
        fig_dt, axes_dt = plt.subplots(3, 4, figsize=(20, 15))
        fig_dt.suptitle(f'BRN Δt by Channel - {label} ({M1_or_M2})', fontsize=16)
        axes_dt = axes_dt.flatten()
        
        dt_min, dt_max = brn_dt_range
        dt_bin_width = config.BRN_DELTA_T_BIN_WIDTH_NS
        dt_bins = np.arange(dt_min, dt_max + dt_bin_width, dt_bin_width)
        
        for i, ch in enumerate(channels_to_analyze):
            ax = axes_dt[i]
            if ch in channel_data and channel_data[ch]['delta_t'].size > 0:
                errs = np.sqrt(len(channel_data[ch]['delta_t']))
                centers = 0.5 * (dt_bins[:-1] + dt_bins[1:])
                # ax.hist(channel_data[ch]['delta_t'], bins=dt_bins, histtype='step', linewidth=1.5)
                ax.errorbar(
                    centers,
                    np.histogram(channel_data[ch]['delta_t'], bins=dt_bins)[0],
                    yerr=errs,
                    fmt='o', markersize=2, capsize=2,
                    alpha=0.7
                )
                ax.set_title(f'Channel {ch}')
                ax.set_xlabel('Δt (ns)')
                ax.set_ylabel('Events')
                # ax.set_yscale('log')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'Ch {ch}\nNo Data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        for i in range(len(channels_to_analyze), len(axes_dt)):
            axes_dt[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename_label}_{M1_or_M2}_brn_delta_t.png')
        plt.close()
        
        # Plot BRN Area Histograms
        fig_area, axes_area = plt.subplots(3, 4, figsize=(20, 15))
        fig_area.suptitle(f'BRN Area by Channel - {label} ({M1_or_M2})', fontsize=16)
        axes_area = axes_area.flatten()
        
        area_range = hist_config['area_range']
        area_bins = hist_config['area_bins']
        area_bin_edges = np.linspace(*area_range, area_bins + 1)
        
        for i, ch in enumerate(channels_to_analyze):
            ax = axes_area[i]
            if ch in channel_data and channel_data[ch]['area'].size > 0:
                errs = np.sqrt(len(channel_data[ch]['area']))
                centers = 0.5 * (area_bin_edges[:-1] + area_bin_edges[1:])
                # ax.hist(channel_data[ch]['area'], bins=area_bin_edges, histtype='step', linewidth=1.5)
                ax.errorbar(
                    centers,
                    np.histogram(channel_data[ch]['area'], bins=area_bin_edges)[0],
                    yerr=errs,
                    fmt='o', markersize=2, capsize=2,
                    alpha=0.7
                )
                ax.set_title(f'Channel {ch}')
                ax.set_xlabel('Area (ADC)')
                ax.set_ylabel('Events')
                # ax.set_yscale('log')
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'Ch {ch}\nNo Data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        for i in range(len(channels_to_analyze), len(axes_area)):
            axes_area[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename_label}_{M1_or_M2}_brn_area.png')
        plt.close()

        # Plot BRN Delta T vs Area Heatmaps
        heatmap_cfg = getattr(config, 'BRN_HIST_CONFIG', {}) or {}
        heatmap_cmap = heatmap_cfg.get('heatmap_cmap', 'viridis')
        heatmap_logscale = bool(heatmap_cfg.get('heatmap_logscale', True))

        fig_heatmap, axes_heatmap = plt.subplots(3, 4, figsize=(20, 15))
        fig_heatmap.suptitle(f'BRN Δt vs Area by Channel - {label} ({M1_or_M2})', fontsize=16)
        axes_heatmap = axes_heatmap.flatten()

        heatmap_artist = None
        active_axes = []
        max_heatmap_count = 0.0

        for ch in channels_to_analyze:
            ch_pairs = np.asarray(channel_data.get(ch, {}).get('delta_t_area', np.empty((0, 2))), dtype=float)
            if ch_pairs.size == 0 and ch in channel_data:
                dt_values = np.asarray(channel_data[ch].get('delta_t', np.array([])), dtype=float)
                area_values = np.asarray(channel_data[ch].get('area', np.array([])), dtype=float)
                if dt_values.size > 0 and dt_values.size == area_values.size:
                    ch_pairs = np.column_stack((dt_values, area_values))
            if ch_pairs.size > 0:
                heatmap_counts, _, _ = np.histogram2d(
                    ch_pairs[:, 0],
                    ch_pairs[:, 1],
                    bins=[dt_bins, area_bin_edges],
                )
                max_heatmap_count = max(max_heatmap_count, float(np.max(heatmap_counts)))

        norm = Normalize(vmin=0.0, vmax=max_heatmap_count if max_heatmap_count > 0 else 1.0)
        if heatmap_logscale and max_heatmap_count > 1.0:
            norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=0.0, vmax=max_heatmap_count, base=10)

        for i, ch in enumerate(channels_to_analyze):
            ax = axes_heatmap[i]
            ch_pairs = np.asarray(channel_data.get(ch, {}).get('delta_t_area', np.empty((0, 2))), dtype=float)
            if ch_pairs.size == 0 and ch in channel_data:
                dt_values = np.asarray(channel_data[ch].get('delta_t', np.array([])), dtype=float)
                area_values = np.asarray(channel_data[ch].get('area', np.array([])), dtype=float)
                if dt_values.size > 0 and dt_values.size == area_values.size:
                    ch_pairs = np.column_stack((dt_values, area_values))

            if ch_pairs.size > 0:
                heatmap_counts, _, _ = np.histogram2d(
                    ch_pairs[:, 0],
                    ch_pairs[:, 1],
                    bins=[dt_bins, area_bin_edges],
                )
                heatmap_artist = ax.pcolormesh(
                    dt_bins,
                    area_bin_edges,
                    heatmap_counts.T,
                    shading='auto',
                    cmap=heatmap_cmap,
                    norm=norm,
                )
                active_axes.append(ax)
                ax.set_title(f'Channel {ch}')
                ax.set_xlabel('Δt (ns)')
                ax.set_ylabel('Area (ADC)')
                ax.set_xlim(brn_dt_range)
                ax.set_ylim(area_range)
                ax.grid(False)
            else:
                ax.text(0.5, 0.5, f'Ch {ch}\nNo Data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

        for i in range(len(channels_to_analyze), len(axes_heatmap)):
            axes_heatmap[i].axis('off')

        if heatmap_artist is not None and active_axes:
            fig_heatmap.subplots_adjust(left=0.06, right=0.90, bottom=0.07, top=0.92, wspace=0.22, hspace=0.28)
            cax = fig_heatmap.add_axes([0.92, 0.12, 0.018, 0.72])
            colorbar = fig_heatmap.colorbar(heatmap_artist, cax=cax)
            colorbar.set_label('Events')
        else:
            fig_heatmap.subplots_adjust(left=0.06, right=0.96, bottom=0.07, top=0.92, wspace=0.22, hspace=0.28)
        plt.savefig(output_dir / f'{filename_label}_{M1_or_M2}_brn_delta_t_area.png')
        plt.close()

    @staticmethod
    def histogram_brn_channel_data(channel_data, brn_dt_range, hist_config):
        """Convert BRN per-channel arrays into a compact histogram payload."""
        dt_min, dt_max = brn_dt_range
        dt_bin_width = int(getattr(config, 'BRN_DELTA_T_BIN_WIDTH_NS', 128))
        dt_edges = np.arange(dt_min, dt_max + dt_bin_width, dt_bin_width)
        if dt_edges[-1] < dt_max:
            dt_edges = np.append(dt_edges, dt_max)
        elif dt_edges[-1] > dt_max:
            dt_edges[-1] = dt_max

        area_range = tuple(hist_config['area_range'])
        area_bins = int(hist_config['area_bins'])
        area_edges = np.linspace(*area_range, area_bins + 1)

        payload = {
            'delta_t_edges': dt_edges,
            'area_edges': area_edges,
            'counts': {},
        }

        for ch, data in channel_data.items():
            dt_data = np.asarray(data.get('delta_t', np.array([])), dtype=float)
            area_data = np.asarray(data.get('area', np.array([])), dtype=float)
            dt_area_pairs = np.asarray(data.get('delta_t_area', np.empty((0, 2))), dtype=float)
            if dt_area_pairs.ndim == 1 and dt_area_pairs.size == 0:
                dt_area_pairs = np.empty((0, 2), dtype=float)
            elif dt_area_pairs.ndim == 1:
                dt_area_pairs = dt_area_pairs.reshape(-1, 2)

            if dt_area_pairs.size > 0:
                dt_area_counts, _, _ = np.histogram2d(
                    dt_area_pairs[:, 0],
                    dt_area_pairs[:, 1],
                    bins=[dt_edges, area_edges],
                )
            elif dt_data.size > 0 and dt_data.size == area_data.size:
                dt_area_counts, _, _ = np.histogram2d(dt_data, area_data, bins=[dt_edges, area_edges])
            else:
                dt_area_counts = np.zeros((len(dt_edges) - 1, len(area_edges) - 1), dtype=float)

            payload['counts'][ch] = {
                'delta_t': np.histogram(dt_data, bins=dt_edges)[0],
                'area': np.histogram(area_data, bins=area_edges)[0],
                'delta_t_area': dt_area_counts,
            }

        return payload

class Plotter:
    """Handles all plotting operations."""
    
    def __init__(self):
        self.file_handler = FileHandler()
        self.hist_calc = HistogramCalculator()

    def plot_histogram(self, arrays, labels, bins, img_path, title, xlabel,
                       M1_or_M2, logscale=True, figsize=(10, 6)):
        """Plot one or more datasets as overlapping histograms with Poissonic error bars."""
        plt.figure(figsize=figsize)
        outputs = []

        # Normalize `bins` to explicit EDGES once, using pooled non-empty data to set range.
        nonempty = [a for a in arrays if a is not None and getattr(a, "size", 0) > 0]
        if len(nonempty) == 0:
            edges_final = np.array([0.0, 1.0])
        else:
            data_all = np.concatenate(nonempty)
            lo, hi = np.nanmin(data_all), np.nanmax(data_all)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = 0.0, 1.0
            if isinstance(bins, (int, np.integer, str)) or (np.asarray(bins).ndim != 1):
                edges_final = self.hist_calc.bin_edges_from_spec(bins, data_all, (lo, hi))
            else:
                edges_final = np.asarray(bins)

        for data, label in zip(arrays, labels):
            if data is not None and getattr(data, "size", 0) > 0:
                counts, edges = np.histogram(data, bins=edges_final)
                bin_centers = 0.5 * (edges[:-1] + edges[1:])
                errors = np.sqrt(counts)
                
                plt.errorbar(
                    bin_centers, counts, yerr=errors,
                    fmt='o', markersize=4, capsize=3,
                    label=f"{label} (N={len(data)})",
                    alpha=0.7
                )
                outputs.append((counts, edges))
            else:
                outputs.append((np.zeros(len(edges_final) - 1), edges_final))

        plt.xlabel(xlabel)
        plt.ylabel('Events')
        plt.title(f"{title} ({M1_or_M2})")
        if logscale:
            plt.yscale('log')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
        plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
        plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
        plt.tight_layout()
        plt.savefig(img_path)

        # Save histogram data as pickle
        pkl_path = img_path.with_suffix('.pkl')
        if outputs:
            edges0 = outputs[0][1]
            centers = 0.5 * (edges0[:-1] + edges0[1:])
            pickle_data = {
                'centers': centers,
                'histograms': {label: counts for label, (counts, _) in zip(labels, outputs)},
                'errors': {label: np.sqrt(counts) for label, (counts, _) in zip(labels, outputs)}
            }
            self.file_handler.save_pickle(pickle_data, pkl_path)
        plt.close()
        return outputs

    def plot_veto_efficiency(self, trig2_pe, trig2_or_34_pe, bins, vetorange, pe_range, 
                           img_path, pkl_path, title, M1_or_M2):
        """Calculates and plots veto efficiency as a function of total photoelectrons."""
        if trig2_or_34_pe.size == 0:
            print(f"No events for veto efficiency calculation for {title}. Skipping.")
            return {
                'average_efficiency': np.nan,
                'average_efficiency_error': np.nan,
                'valid_bin_count': 0,
                'total_trig2': int(trig2_pe.size),
                'total_trig2_or_34': int(trig2_or_34_pe.size)
            }

        # Use vetorange for binning to ensure bins align with display range
        veto_min, veto_max = vetorange
        bin_edges = np.linspace(veto_min, veto_max, bins + 1)
        
        counts_2, _ = np.histogram(trig2_pe, bins=bin_edges)
        counts_2_or_34, _ = np.histogram(trig2_or_34_pe, bins=bin_edges)
        
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Calculate Efficiency and Error
        efficiency = np.zeros_like(counts_2, dtype=float)
        error = np.zeros_like(counts_2, dtype=float)
        valid_mask = counts_2_or_34 > 0
        
        ratio = np.divide(counts_2[valid_mask], counts_2_or_34[valid_mask])
        efficiency[valid_mask] = 1 - ratio
        avg_mask = valid_mask & (bin_centers >= vetorange[0]) & (bin_centers <= vetorange[1])
        average_efficiency = np.mean(efficiency[avg_mask]) if np.any(avg_mask) else np.nan
        
        n = counts_2_or_34[valid_mask]
        p = ratio
        error[valid_mask] = np.sqrt(p * (1 - p) / n)
        # Run-average uncertainty should come from integrated counts in the veto range,
        # not RMS of per-bin errors.
        k_total = float(np.sum(counts_2[avg_mask])) if np.any(avg_mask) else 0.0
        n_total = float(np.sum(counts_2_or_34[avg_mask])) if np.any(avg_mask) else 0.0
        if n_total > 0:
            p_total = np.clip(k_total / n_total, 0.0, 1.0)
            average_efficiency = 1.0 - p_total
            average_efficiency_error = np.sqrt(p_total * (1.0 - p_total) / n_total)
        else:
            average_efficiency = np.nan
            average_efficiency_error = np.nan

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(bin_centers[valid_mask], efficiency[valid_mask], yerr=error[valid_mask],
                     fmt='o', capsize=3, label='efficiency = 1 - N(trig=2) / N(trig=2 or 34)', 
                     color='navy', markersize=5)
        plt.axhline(average_efficiency, color='red', linestyle='--',
                    label=f'Average Efficiency = {average_efficiency:.4f}')
        plt.xlabel('Total Photoelectrons (P.E.)')
        plt.ylabel('Veto Efficiency')
        plt.title(f"{title} ({M1_or_M2})")
        plt.xlim(vetorange)
        plt.ylim(0.99, 1.005)
        plt.grid(which='major', linestyle='-', linewidth=0.7)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.legend()
        self.file_handler.ensure_dir(img_path.parent)
        plt.savefig(img_path)
        plt.close()

        # Save Data
        pickle_data = {
            'centers': bin_centers, 'efficiency': efficiency, 'error': error,
            'counts_2': counts_2, 'counts_2_or_34': counts_2_or_34
        }
        self.file_handler.save_pickle(pickle_data, pkl_path)
        print(f"Veto efficiency plot saved to {img_path}")
        print(f"Veto efficiency data saved to {pkl_path}")

        return {
            'average_efficiency': float(average_efficiency) if np.isfinite(average_efficiency) else np.nan,
            'average_efficiency_error': float(average_efficiency_error) if np.isfinite(average_efficiency_error) else np.nan,
            'valid_bin_count': int(np.count_nonzero(avg_mask)),
            'total_trig2': int(trig2_pe.size),
            'total_trig2_or_34': int(trig2_or_34_pe.size)
        }

    def plot_correlation_maps(self, df, output_dir, label, M1_or_M2):
        """Plots a 3x3 grid of correlation maps for delta_t, total_pe, and multiplicity."""
        self.file_handler.ensure_dir(output_dir)
        if df.empty:
            print(f"DataFrame is empty for {label}. Skipping correlation map.")
            return

        variables = ['delta_t', 'total_pe', 'multiplicity']
        pretty_labels = ['Δt (ns)', 'Total Photoelectrons', 'Multiplicity']

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'Correlation Matrix ({label}, {M1_or_M2})', fontsize=18)

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                var_y = variables[i]
                var_x = variables[j]

                if i == 2: ax.set_xlabel(pretty_labels[j], fontsize=12)
                if j == 0: ax.set_ylabel(pretty_labels[i], fontsize=12)

                if i == j:
                    data = df[var_x].dropna()
                    if not data.empty:
                        ax.hist(data, bins=50, histtype='step', linewidth=1.5, color='k')
                    ax.set_yscale('log')
                    ax.grid(True, which='both', linestyle=':')
                else:
                    subset = df[[var_x, var_y]].dropna()
                    if not subset.empty and len(subset) > 1:
                        h = ax.hist2d(subset[var_x], subset[var_y],
                                      bins=50, cmap='viridis', norm=LogNorm())
                        if h[0].max() > 0: 
                            fig.colorbar(h[3], ax=ax)

                        corr, _ = pearsonr(subset[var_x], subset[var_y])
                        corr_text = f'Corr: {corr:.2f}'
                        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

                if i < 2: ax.tick_params(axis='x', labelbottom=False)
                if j > 0: ax.tick_params(axis='y', labelleft=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
        save_path = output_dir / f'{filename_label}_{M1_or_M2}_correlation_map.png'
        
        plt.savefig(save_path)
        print(f"Correlation map saved to {save_path}")
        plt.close()

class RunProcessor:
    """Main class for processing individual runs."""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.plotter = Plotter()
        self.file_handler = FileHandler()

    def _resolve_input_file(self, run, data_dir, M1_or_M2):
        """Resolve input ROOT file path, including optional M1 auto-suffix fallback."""
        if M1_or_M2 == 'M1':
            raw_suffix = getattr(config, 'suffix_M1', '_processed_v5.root')
            if isinstance(raw_suffix, str) and raw_suffix.lower() == 'auto':
                suffix_candidates = list(getattr(
                    config,
                    'SUFFIX_M1_CANDIDATES',
                    ['_processed_v5.root', '_processed_v4.root']
                ))
            else:
                suffix_candidates = [raw_suffix]

            seen = set()
            for suffix in suffix_candidates:
                if suffix in seen:
                    continue
                seen.add(suffix)
                infile = data_dir / f"run{run}{suffix}"
                if infile.exists():
                    return infile, suffix

            return None, suffix_candidates

        if M1_or_M2 == 'M2':
            suffix = getattr(config, 'suffix_M2', '_processed_H2O_v5.root')
            infile = data_dir / f"run{run}{suffix}"
            return (infile, suffix) if infile.exists() else (None, [suffix])

        raise ValueError("M1_or_M2 must be 'M1' or 'M2'")

    def process_run(self, run, data_dir, output_dir, delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                    multiplicity_spe, multiplicity_cut, time_std_cut, logscale,
                    low_light_fit_range, simp_hist_config, M1_or_M2):
        """Process a single run: read data, perform calculations, and apply cuts."""
        print(f"--- Processing run {run} ---")
        
        infile, used_suffix = self._resolve_input_file(run, data_dir, M1_or_M2)
        if infile is None:
            print(f"Missing file for run {run}. Tried suffix(es): {used_suffix}")
            return None
        print(f"Using input file: {infile.name} (suffix={used_suffix})")

        # Get run start time
        run_start_time_str = self._get_run_start_time(infile, run)
        
        # Read data
        df_all = self._read_root_file(infile)
        if df_all is None:
            return None

        # Setup output directories
        run_dir = output_dir / f"run{run}_{run_start_time_str}"
        hist_dir = run_dir / "histograms"
        cut_dir = run_dir / "cuthist"
        ll_dir = run_dir / "lowlight"
        sipm_fit_dir = run_dir / "sipm_pulseh_fit"
        
        for dir_path in [hist_dir, cut_dir, ll_dir, sipm_fit_dir]:
            self.file_handler.ensure_dir(dir_path)

        # Calculate time length and save to file
        timelength = 0.0
        if 'nsTime' in df_all.columns and not df_all.empty:
            time_values = df_all['nsTime'].values
            timelength = float(np.max(time_values) - np.min(time_values))
            length_seconds = timelength / 1e9
            length_min = timelength / 1e9 / 60
            print("Length of time:", timelength, "ns")
            print("Length of time:", length_min, "minutes")
            
            time_data = {
                "timelength_ns": timelength,
                "timelength_s": length_seconds,
                "timelength_min": length_min
            }
            with open(run_dir / "time_length.json", "w") as f:
                json.dump(time_data, f, indent=4)

        # Process data and generate outputs
        result = self._process_run_data(df_all, run, run_dir, hist_dir, cut_dir, ll_dir, sipm_fit_dir,
                                      delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                                      multiplicity_spe, multiplicity_cut, time_std_cut,
                          logscale, low_light_fit_range, M1_or_M2, run_start_time_str)
        
        return result, timelength

    def _get_run_start_time(self, infile, run):
        """Extract run start time from ROOT file."""
        try:
            with uproot.open(infile) as f_ts:
                if 'starttime' in f_ts:
                    unix_time = f_ts['starttime'].member("fVal")
                    return datetime.fromtimestamp(unix_time).strftime('%Y%m%d-%H')
        except Exception as e:
            print(f"Warning: Could not read start time for run {run}, using default folder name. Error: {e}")
        return "no_ts"

    def _read_root_file(self, infile):
        """Read ROOT file and return DataFrame."""
        dfs = []
        branches = ['eventID', 'nsTime', 'triggerBits', 'area', 'peakPosition', 'pulseH']
        
        try:
            for chunk in uproot.open(infile)['tree'].iterate(branches, library='ak', step_size='500 MB'):
                df = pd.DataFrame({
                    'eventID': ak.to_numpy(chunk['eventID']),
                    'nsTime': ak.to_numpy(chunk['nsTime']),
                    'triggerBits': ak.to_numpy(chunk['triggerBits']),
                    'area_array': ak.to_list(chunk['area']),
                    'peakPosition': ak.to_list(chunk['peakPosition']),
                    'pulseH_array': ak.to_list(chunk['pulseH']),
                })
                dfs.append(df)
        except uproot.KeyInFileError:
            print(f"Warning: 'pulseH' branch not found in {infile}. Reading without it.")
            branches.remove('pulseH')
            for chunk in uproot.open(infile)['tree'].iterate(branches, library='ak', step_size='500 MB'):
                df = pd.DataFrame({
                    'eventID': ak.to_numpy(chunk['eventID']),
                    'nsTime': ak.to_numpy(chunk['nsTime']),
                    'triggerBits': ak.to_numpy(chunk['triggerBits']),
                    'area_array': ak.to_list(chunk['area']),
                    'peakPosition': ak.to_list(chunk['peakPosition']),
                })
                dfs.append(df)

        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def _process_run_data(self, df_all, run, run_dir, hist_dir, cut_dir, ll_dir, sipm_fit_dir,
                         delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                         multiplicity_spe, multiplicity_cut, time_std_cut,
                         logscale, low_light_fit_range, M1_or_M2, run_start_time_str):
        """Process the data for a single run."""
        
        # Plot trigger bits
        self.plotter.plot_histogram(
            [df_all['triggerBits'].to_numpy()], ['triggerBits'],
            np.arange(0, 37), hist_dir / f"{run}_{M1_or_M2}_triggerBits.png",
            f"Run {run} Trigger Bits", "Trigger Bits", M1_or_M2, logscale
        )
        
        # Process low-light events and calculate photoelectrons
        ll_events = df_all[df_all['triggerBits'] == 16]
        mu1_values_run, mu1_errors_run, ll_hist_counts, ll_bin_edges = self._process_low_light_events(
            ll_events, ll_dir, run, M1_or_M2, low_light_fit_range
        )

        # Highlight events (triggerBits == 8): PMT PE spectra and Gaussian peak extraction
        highlight_dir = run_dir / "highlight"
        self.file_handler.ensure_dir(highlight_dir)
        highlight_events = df_all[df_all['triggerBits'] == 8]
        highlight_fit_config = self._get_default_highlight_fit_config()
        hl_hist_counts, hl_bin_edges, hl_summary_df, hl_sum_payload = self._fit_and_plot_highlight_pe(
            highlight_events,
            mu1_values_run,
            mu1_errors_run,
            highlight_dir,
            f'Run{run}',
            M1_or_M2,
            highlight_fit_config
        )
        
        # Calculate derived quantities
        df_all = self._calculate_derived_quantities(df_all, mu1_values_run, multiplicity_spe)
        # Check if the cut is enabled (set to > 0)
        if config.TIME_INTERVAL_CUT_NS > 0:
            print(f"Applying {config.TIME_INTERVAL_CUT_NS} ns time interval cut...")
            original_event_count = len(df_all)
            
            # Calculate the time difference from the previous event
            time_diff_ns = df_all['nsTime'].diff()
            
            # Create a mask to KEEP events.
            # We keep an event if:
            # 1. The time diff is >= the cut
            # 2. The time diff is NaN (this is the first event, which we explicitly keep)
            time_interval_mask = (time_diff_ns >= config.TIME_INTERVAL_CUT_NS) | (time_diff_ns.isna())
            
            # Apply the mask to the main DataFrame
            df_all = df_all[time_interval_mask].reset_index(drop=True)
            
            print(f"Time interval cut: Kept {len(df_all)} / {original_event_count} events")
        # >>> END: NEW TIME INTERVAL CUT <<<
        
        # Save processed data
        df_all.to_pickle(run_dir / f"run{run}_{M1_or_M2}_data_with_pe.pkl")
        
        # Apply cuts and generate veto efficiency plots
        cut_payload, pe_trig2, pe_trig2_or_34, veto_summary, michel_summary = self._apply_cuts_and_generate_plots(
            df_all, run, hist_dir, cut_dir, delta_t_cut, pe_cut, bins, veto_bins,
            vetorange, multiplicity_cut, time_std_cut, logscale, M1_or_M2
        )
        veto_hist_payload = self._build_veto_hist_payload(
            pe_trig2.to_numpy(), pe_trig2_or_34.to_numpy(), bins, veto_bins, vetorange, pe_cut
        )

        event61_fit_config = self._get_default_event61_fit_config()
        event61_summary = _empty_event61_analysis(
            getattr(config, 'EVENT61_CHANNEL_INDEX', 22),
            False,
            event61_fit_config,
            n_entries=0,
        )
        event61_hist_payload = None
        full_area_array = None
        pulseh_array = None
        brn_trigger_info = None
        needs_pulseh_array = (
            event61_fit_config.get('enabled', True)
            or config.PERFORM_THIN_VETO_ANALYSIS
            or config.PERFORM_BRN_ANALYSIS
        )
        if 'pulseH_array' in df_all.columns and needs_pulseh_array:
            pulseh_array = np.array(df_all['pulseH_array'].to_list())
            if 'area_array' in df_all.columns and (config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS):
                full_area_array = np.array(df_all['area_array'].to_list())
            if config.PERFORM_BRN_ANALYSIS:
                brn_trigger_info = BRNAnalyzer.build_brn_trigger_info(df_all['triggerBits'].to_numpy(), pulseh_array)

        if event61_fit_config.get('enabled', True):
            event61_dir = run_dir / 'event61'
            self.file_handler.ensure_dir(event61_dir)
            if pulseh_array is not None:
                event61_hist_payload = build_event61_histogram_payload(
                    pulseh_array,
                    fit_config=event61_fit_config,
                    channel_index=getattr(config, 'EVENT61_CHANNEL_INDEX', 22),
                )
                event61_summary = plot_event61_histogram_payload(
                    event61_hist_payload,
                    event61_dir,
                    f'Run{run}',
                    M1_or_M2,
                    fit_config=event61_fit_config,
                    filename_suffix='event61_pulseh_fit',
                    title_prefix='Event61 pulseH',
                )
            else:
                print(f"No pulseH data available for run {run}; skipping Event61 histogram fit.")

        beam_on_count = int(np.count_nonzero(df_all['triggerBits'].to_numpy() == 1))
        brn_beam_on_count = beam_on_count
        event61_adjustment_applied = False
        event61_adc_min, event61_adc_max = _get_event61_adc_window()
        event61_channel_index = int(getattr(config, 'EVENT61_CHANNEL_INDEX', 22))
        event61_count = 0
        event61_channel_available = False
        if brn_trigger_info is not None:
            brn_beam_on_count = int(np.count_nonzero(brn_trigger_info['beam_on_mask']))
            event61_adjustment_applied = bool(brn_trigger_info['event61_adjustment_applied'])
            event61_adc_min = float(brn_trigger_info['event61_adc_min'])
            event61_adc_max = float(brn_trigger_info['event61_adc_max'])
            event61_channel_index = int(brn_trigger_info['event61_channel_index'])
            event61_count = int(brn_trigger_info['event61_count'])
            event61_channel_available = bool(brn_trigger_info['event61_channel_available'])

        run_veto_summary = {
            'run': int(run),
            'run_start_time': run_start_time_str,
            'beam_on_count': beam_on_count,
            'brn_beam_on_count': brn_beam_on_count,
            'average_efficiency': veto_summary.get('average_efficiency', np.nan),
            'average_efficiency_error': veto_summary.get('average_efficiency_error', np.nan),
            'valid_bin_count': veto_summary.get('valid_bin_count', 0),
            'total_trig2': veto_summary.get('total_trig2', 0),
            'total_trig2_or_34': veto_summary.get('total_trig2_or_34', 0),
            'mu1_values': [float(x) if np.isfinite(x) else np.nan for x in np.asarray(mu1_values_run, dtype=float)],
            'mu1_errors': [float(x) if np.isfinite(x) else np.nan for x in np.asarray(mu1_errors_run, dtype=float)],
            'michel_peak_pe': float(michel_summary.get('peak_location', np.nan)),
            'michel_peak_pe_err': float(michel_summary.get('peak_location_error', np.nan)),
            'michel_sigma_pe': float(michel_summary.get('sigma', np.nan)),
            'michel_sigma_pe_err': float(michel_summary.get('sigma_error', np.nan)),
            'michel_fwhm_range': [
                float(michel_summary.get('fwhm_min', np.nan)),
                float(michel_summary.get('fwhm_max', np.nan))
            ],
            'michel_fit_success': bool(michel_summary.get('success', False)),
            'event61_adjustment_applied': event61_adjustment_applied,
            'event61_adc_range': _serialize_event61_adc_window(event61_adc_min, event61_adc_max),
            'event61_threshold_adc': event61_adc_min,
            'event61_channel_index': event61_channel_index,
            'event61_count': event61_count,
            'event61_channel_available': event61_channel_available,
            'event61_hist_total_entries': int(event61_summary.get('n_entries', 0)),
            'event61_fit_success': bool(event61_summary.get('fit_success', False)),
            'event61_fit_mean_adc': float(event61_summary.get('fit_mean_adc', np.nan)),
            'event61_fit_mean_adc_err': float(event61_summary.get('fit_mean_adc_err', np.nan)),
            'event61_fit_sigma_adc': float(event61_summary.get('fit_sigma_adc', np.nan)),
            'event61_fit_sigma_adc_err': float(event61_summary.get('fit_sigma_adc_err', np.nan)),
            'event61_fit_constant': float(event61_summary.get('fit_constant', np.nan)),
            'event61_fit_constant_err': float(event61_summary.get('fit_constant_err', np.nan)),
            'event61_fit_reduced_chi2': float(event61_summary.get('fit_reduced_chi2', np.nan)),
            'event61_raw_window_count': float(event61_summary.get('raw_window_count', 0.0)),
            'event61_background_window_count': float(event61_summary.get('background_window_count', np.nan)),
            'event61_background_window_count_err': float(event61_summary.get('background_window_count_err', np.nan)),
            'event61_background_subtracted_count': float(event61_summary.get('background_subtracted_count', np.nan)),
            'event61_background_subtracted_count_err': float(event61_summary.get('background_subtracted_count_err', np.nan)),
        }

        hl_peak = np.full(12, np.nan)
        hl_peak_err = np.full(12, np.nan)
        if hl_summary_df is not None and not hl_summary_df.empty:
            for _, row in hl_summary_df.iterrows():
                ch = int(row['channel'])
                if 0 <= ch < 12:
                    peak_val = row.get('peak_pe', np.nan)
                    peak_err_val = row.get('peak_pe_err', np.nan)
                    hl_peak[ch] = float(peak_val) if pd.notna(peak_val) else np.nan
                    hl_peak_err[ch] = float(peak_err_val) if pd.notna(peak_err_val) else np.nan

        finite_hl = hl_peak[np.isfinite(hl_peak)]
        run_veto_summary['highlight_peak_pe'] = [float(x) if np.isfinite(x) else np.nan for x in hl_peak]
        run_veto_summary['highlight_peak_pe_err'] = [float(x) if np.isfinite(x) else np.nan for x in hl_peak_err]
        run_veto_summary['highlight_avg_pe'] = float(np.mean(finite_hl)) if finite_hl.size > 0 else np.nan

        with open(run_dir / "run_veto_summary.json", "w") as f:
            json.dump(run_veto_summary, f, indent=4)
        
        # Extract SiPM events (triggerBits >= 32)
        sipm_events_df = df_all[df_all['triggerBits'] >= 32]
        sipm_hist_payload = self._build_sipm_area_hist_payload(sipm_events_df)

        sipm_pulseh_fit_config = self._get_default_sipm_pulseh_fit_config()
        sipm_fit_results = {}
        if sipm_pulseh_fit_config.get('enabled', True):
            if 'pulseH_array' in sipm_events_df.columns and not sipm_events_df.empty:
                sipm_fit_results = self._fit_and_plot_sipm_pulseh(
                    sipm_events_df,
                    sipm_fit_dir,
                    f"Run{run}",
                    M1_or_M2,
                    sipm_pulseh_fit_config
                )
            else:
                print(f"No SiPM pulseH data available for run {run}; skipping SiPM pulseH fit.")
        
        # Initialize thin veto and BRN data
        tv_hist_counts = {}
        tv_bin_edges = {}
        brn_hist_payload = None
        
        # Perform thin veto and BRN analysis if pulseH data is available
        if pulseh_array is not None and (config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS):
            # Thin veto analysis
            if config.PERFORM_THIN_VETO_ANALYSIS:
                tv_raw_data = ThinVetoAnalyzer.plot_thin_veto_performance(
                    df_all, pulseh_array, full_area_array, hist_dir, f"Run {run}", M1_or_M2,
                    config.THIN_VETO_CHANNELS, config.THIN_VETO_THRESHOLD,
                    config.MULTIPLICITY_CUT, config.THIN_VETO_HIST_CONFIG
                )
                
                if tv_raw_data:
                    tv_muon_h, tv_muon_a, tv_no_co_h, tv_no_co_a = tv_raw_data
                    tv_data_dict = {
                        'muon_h': tv_muon_h, 'muon_a': tv_muon_a,
                        'no_co_h': tv_no_co_h, 'no_co_a': tv_no_co_a
                    }
                    tv_hist_counts, tv_bin_edges = self.plotter.hist_calc.calculate_histograms(
                        tv_data_dict, config.THIN_VETO_HIST_CONFIG
                    )
                else:
                    keys = ['muon_h', 'muon_a', 'no_co_h', 'no_co_a']
                    tv_hist_counts, tv_bin_edges = self.plotter.hist_calc.calculate_histograms(
                        {k: None for k in keys}, config.THIN_VETO_HIST_CONFIG
                    )
            
            # BRN analysis
            if config.PERFORM_BRN_ANALYSIS:
                brn_data = BRNAnalyzer.compute_brn_data(
                    df_all, pulseh_array, full_area_array,
                    config.BRN_SIPM_CHANNELS,
                    config.BRN_SIPM_THRESHOLD_ADC,
                    brn_trigger_info=brn_trigger_info,
                )
                if brn_data:
                    BRNAnalyzer.plot_brn_histograms(
                        brn_data, hist_dir, f"Run {run}", M1_or_M2,
                        config.BRN_DELTA_T_RANGE, config.BRN_HIST_CONFIG
                    )
                    brn_hist_payload = BRNAnalyzer.histogram_brn_channel_data(
                        brn_data, config.BRN_DELTA_T_RANGE, config.BRN_HIST_CONFIG
                    )
        else:
            if config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS:
                print(f"Warning: 'pulseH_array' not found for run {run}. Skipping thin veto and BRN analysis.")
            keys = ['muon_h', 'muon_a', 'no_co_h', 'no_co_a']
            tv_hist_counts, tv_bin_edges = self.plotter.hist_calc.calculate_histograms(
                {k: None for k in keys}, config.THIN_VETO_HIST_CONFIG
            )
        
        # Return all processed data
        if cut_payload:
            return (
                cut_payload,
                ll_hist_counts, ll_bin_edges,
                hl_hist_counts, hl_bin_edges, hl_sum_payload,
                event61_hist_payload,
                sipm_hist_payload, veto_hist_payload,
                tv_hist_counts, tv_bin_edges,
                brn_hist_payload,
                sipm_fit_results,
                run_veto_summary,
            )
        return (
            None,
            ll_hist_counts, ll_bin_edges,
            hl_hist_counts, hl_bin_edges, hl_sum_payload,
            event61_hist_payload,
            sipm_hist_payload, veto_hist_payload,
            tv_hist_counts, tv_bin_edges,
            brn_hist_payload,
            sipm_fit_results,
            run_veto_summary,
        )

    def _get_default_sipm_pulseh_fit_config(self):
        cfg = getattr(config, 'SIPM_PULSEH_FIT_CONFIG', {}) or {}
        default_hist_max = cfg.get('hist_range', (0, 800))[1] if isinstance(cfg.get('hist_range', (0, 800)), (tuple, list)) else 800
        return {
            'enabled': cfg.get('enabled', True),
            'bins': int(cfg.get('bins', 200)),
            'hist_range': tuple(cfg.get('hist_range', (0, 800))),
            'threshold': float(cfg.get('threshold', 25.0)),
            'fit_ranges_by_panel': cfg.get('fit_ranges_by_panel', {
                'top': (120, default_hist_max),
                'wide': (160, default_hist_max),
                'thin': (90, default_hist_max),
            }),
            'mpv_bounds_by_panel': cfg.get('mpv_bounds_by_panel', {
                'top': (120, 300),
                'wide': (160, 400),
                'thin': (90, 300),
            })
        }

    def _get_default_highlight_fit_config(self):
        cfg = getattr(config, 'HIGHLIGHT_FIT_CONFIG', {}) or {}
        return {
            'bins': int(cfg.get('bins', 120)),
            'hist_range': tuple(cfg.get('hist_range', (0, 120))),
            'sum_bins': int(cfg.get('sum_bins', cfg.get('bins', 120))),
            'sum_hist_range': tuple(cfg.get('sum_hist_range', (0, 1200))),
            'fit_window_half_width_pe': float(cfg.get('fit_window_half_width_pe', 12.0)),
            'min_fit_points': int(cfg.get('min_fit_points', 6)),
        }

    def _get_default_event61_fit_config(self):
        return get_event61_fit_config()

    def _build_sipm_area_hist_payload(self, sipm_events_df):
        """Build subjob-level SiPM area histograms without storing raw event frames."""
        hist_cfg = getattr(config, 'SIPM_HIST_CONFIG', {}) or {}
        hist_bins = int(hist_cfg.get('hist_bins', 100))
        hist_range = tuple(hist_cfg.get('hist_range', (-50, 4000)))
        edges = np.linspace(*hist_range, hist_bins + 1)
        counts = {ch: np.zeros(hist_bins, dtype=float) for ch in config.SIPM_CHANNELS}

        if sipm_events_df.empty or 'area_array' not in sipm_events_df.columns:
            return {'edges': edges, 'counts': counts}

        area_data = np.array(sipm_events_df['area_array'].to_list())
        if area_data.ndim != 2:
            return {'edges': edges, 'counts': counts}

        for ch in config.SIPM_CHANNELS:
            if ch < area_data.shape[1]:
                counts[ch], _ = np.histogram(area_data[:, ch], bins=edges)

        return {'edges': edges, 'counts': counts}

    def _build_veto_hist_payload(self, pe_trig2, pe_trig2_or_34, bins, veto_bins, vetorange, pe_range):
        """Build master-ready veto histogram payloads at subjob time."""
        trig2 = np.asarray(pe_trig2, dtype=float)
        trig2_or_34 = np.asarray(pe_trig2_or_34, dtype=float)
        if trig2.size > 0 or trig2_or_34.size > 0:
            comparison_data = np.concatenate([arr for arr in (trig2, trig2_or_34) if arr.size > 0])
        else:
            comparison_data = np.array([], dtype=float)

        comparison_edges = self.plotter.hist_calc.bin_edges_from_spec(bins, comparison_data, pe_range)
        efficiency_edges = np.linspace(vetorange[0] * 0.5, vetorange[1], veto_bins + 1)

        comparison_counts_2, _ = np.histogram(trig2, bins=comparison_edges)
        comparison_counts_2_or_34, _ = np.histogram(trig2_or_34, bins=comparison_edges)
        efficiency_counts_2, _ = np.histogram(trig2, bins=efficiency_edges)
        efficiency_counts_2_or_34, _ = np.histogram(trig2_or_34, bins=efficiency_edges)

        return {
            'comparison_edges': comparison_edges,
            'comparison_counts_2': comparison_counts_2,
            'comparison_counts_2_or_34': comparison_counts_2_or_34,
            'efficiency_edges': efficiency_edges,
            'efficiency_counts_2': efficiency_counts_2,
            'efficiency_counts_2_or_34': efficiency_counts_2_or_34,
        }

    @staticmethod
    def _landau_plus_exp(x, A, mpv, sigma, tau, B, x0, C):
        return A * landau_dist.pdf(x, loc=mpv, scale=sigma) + B * np.exp(-(x - x0) / tau) + C

    @staticmethod
    def _detection_probability_from_fit(mpv, sigma, threshold):
        denom = 1.0 - landau_dist.cdf(0, loc=mpv, scale=sigma)
        if denom <= 0:
            return np.nan
        return float((1.0 - landau_dist.cdf(threshold, loc=mpv, scale=sigma)) / denom)

    @staticmethod
    def _extract_channel_data(df_muon, j):
        arr = np.array(df_muon['pulseH_array'].to_list())
        return arr[:, j].astype(float)

    @staticmethod
    def _gaussian(x, amp, mu, sigma, c):
        sigma = np.maximum(sigma, 1e-6)
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

    def _fit_one_sipm_channel_hist(self, data, bins, hist_range, fit_range, mpv_bounds):
        counts, edges = np.histogram(data, bins=bins, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]

        xfit_min, xfit_max = fit_range
        mask = (centers >= xfit_min) & (centers <= xfit_max)
        x = centers[mask]
        y = counts[mask]

        good = y > 0
        x = x[good]
        y = y[good]
        if len(x) < 8:
            raise RuntimeError("Too few non-zero bins in fit window")

        C0 = 0.0
        A0 = float(max(1.0, (np.max(y) - C0) * (xfit_max - xfit_min)))
        mpv0 = float(x[np.argmax(y)])
        sigma0 = float(max(1e-3, 0.08 * (xfit_max - xfit_min)))
        tau0 = float(max(1.0, 0.5 * (xfit_max - xfit_min)))
        B0 = float(max(1.0, 0.1 * A0 / max(tau0, 1e-3)))
        x00 = float(mpv0)

        p0 = [A0, mpv0, sigma0, tau0, B0, x00, C0]
        bounds = (
            [0.0, mpv_bounds[0], 1e-6, 1e-6, 0.0, hist_range[0], 0.0],
            [np.inf, mpv_bounds[1], np.inf, np.inf, np.inf, hist_range[1], np.inf]
        )

        sigma_y = np.sqrt(y)
        popt, pcov = curve_fit(
            self._landau_plus_exp, x, y,
            p0=p0, bounds=bounds,
            sigma=sigma_y, absolute_sigma=False,
            maxfev=50000
        )
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)

        A, mpv, sigma, tau, B, x0, C = popt
        return {
            'counts': counts,
            'edges': edges,
            'centers': centers,
            'widths': widths,
            'popt': popt,
            'perr': perr,
            'A': float(A),
            'mpv': float(mpv),
            'sigma': float(sigma),
            'tau': float(tau),
            'B': float(B),
            'x0': float(x0),
            'C': float(C),
        }

    def _fit_and_plot_sipm_pulseh(self, sipm_events_df, output_dir, file_label, M1_or_M2, fit_config):
        self.file_handler.ensure_dir(output_dir)

        bins = fit_config['bins']
        hist_range = tuple(fit_config['hist_range'])
        threshold = float(fit_config['threshold'])
        fit_ranges_by_panel = fit_config['fit_ranges_by_panel']
        mpv_bounds_by_panel = fit_config['mpv_bounds_by_panel']

        sipm_channels = list(config.SIPM_CHANNELS)
        fig, axs = plt.subplots(2, 5, figsize=(16, 6))
        fig.suptitle(
            f"{file_label} {M1_or_M2} SiPM pulseH: Landau+Exp fits, bins={bins}, threshold={threshold}",
            fontsize=14
        )

        fit_results_data = {}
        records = []
        shared_handles = None
        shared_labels = None

        for idx, j in enumerate(sipm_channels):
            ch = j - 11
            panel_type = PANEL_GROUP.get(ch, 'top')
            fit_range = tuple(fit_ranges_by_panel.get(panel_type, hist_range))
            mpv_bounds = tuple(mpv_bounds_by_panel.get(panel_type, fit_range))

            r, c = divmod(idx, 5)
            ax = axs[r, c]

            try:
                data = self._extract_channel_data(sipm_events_df, j)
                fit_out = self._fit_one_sipm_channel_hist(data, bins, hist_range, fit_range, mpv_bounds)
                p_det = self._detection_probability_from_fit(fit_out['mpv'], fit_out['sigma'], threshold)

                hist_line = ax.step(fit_out['edges'][:-1], fit_out['counts'], where='post', linewidth=1, label='Hist')
                xx = np.linspace(fit_range[0], fit_range[1], 500)
                fit_line, = ax.plot(
                    xx,
                    self._landau_plus_exp(
                        xx, fit_out['A'], fit_out['mpv'], fit_out['sigma'],
                        fit_out['tau'], fit_out['B'], fit_out['x0'], fit_out['C']
                    ),
                    linewidth=1.5,
                    label='Fit'
                )
                thr_line = ax.axvline(threshold, linestyle='--', linewidth=1, label=f'Thr={threshold:g}')
                fit_span = ax.axvspan(fit_range[0], fit_range[1], color='orange', alpha=0.1, label='Fit Range')

                if shared_handles is None:
                    shared_handles = [hist_line[0], fit_line, thr_line, fit_span]
                    shared_labels = [h.get_label() for h in shared_handles]

                ax.set_title(
                    f"SiPM {ch} \n"
                    f"MPV={fit_out['mpv']:.2f}, σ={fit_out['sigma']:.2f}\n"
                    f"P>{threshold}={p_det:.3f}"
                )
                ax.set_xlim(hist_range)
                ax.set_ylim(8, None)
                ax.set_yscale('log')
                bin_width = float(np.median(fit_out['widths'])) if fit_out['widths'].size > 0 else 0.0
                ax.set_xlabel('Pulse Height (ADC)')
                ax.set_ylabel(f'Counts/{bin_width:.1f} ADC')
                ax.grid(True, which='both', linestyle='--', alpha=0.4)

                ax.set_xticks(np.linspace(hist_range[0], hist_range[1], 5))
                ax.set_xticks(np.linspace(hist_range[0], hist_range[1], 21), minor=True)
                ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5], minor=False)

                fit_results_data[ch] = {
                    'channel': ch,
                    'panel': panel_type,
                    'counts': fit_out['counts'],
                    'edges': fit_out['edges'],
                    'centers': fit_out['centers'],
                    'widths': fit_out['widths'],
                    'fit_range': fit_range,
                    'mpv_bounds': mpv_bounds,
                    'popt': fit_out['popt'],
                    'perr': fit_out['perr'],
                    'A': fit_out['A'],
                    'mpv': fit_out['mpv'],
                    'sigma': fit_out['sigma'],
                    'tau': fit_out['tau'],
                    'B': fit_out['B'],
                    'x0': fit_out['x0'],
                    'C': fit_out['C'],
                    'threshold': threshold,
                    'p_detect': p_det,
                    'landau_backend': 'landau' if HAVE_TRUE_LANDAU else 'moyal'
                }
                records.append({
                    'channel': ch,
                    'panel': panel_type,
                    'A': fit_out['A'],
                    'mpv': fit_out['mpv'],
                    'sigma': fit_out['sigma'],
                    'tau': fit_out['tau'],
                    'B': fit_out['B'],
                    'x0': fit_out['x0'],
                    'C': fit_out['C'],
                    'threshold': threshold,
                    'p_detect': p_det,
                })
            except Exception as e:
                ax.set_title(f"SiPM {ch} FAILED")
                ax.text(0.05, 0.5, str(e), transform=ax.transAxes, fontsize=8)
                ax.set_xlim(hist_range)
                fail_bin_width = float((hist_range[1] - hist_range[0]) / bins) if bins > 0 else 0.0
                ax.set_xlabel('Pulse Height (ADC)')
                ax.set_ylabel(f'Counts/{fail_bin_width:.1f} ADC')
                ax.grid(True, linestyle='--', alpha=0.4)

                fit_results_data[ch] = {
                    'channel': ch,
                    'panel': panel_type,
                    'counts': np.zeros(bins),
                    'edges': np.linspace(hist_range[0], hist_range[1], bins + 1),
                    'fit_range': fit_range,
                    'mpv_bounds': mpv_bounds,
                    'popt': None,
                    'perr': None,
                    'threshold': threshold,
                    'p_detect': np.nan,
                    'fit_error': str(e),
                    'landau_backend': 'landau' if HAVE_TRUE_LANDAU else 'moyal'
                }
                records.append({
                    'channel': ch,
                    'panel': panel_type,
                    'A': np.nan,
                    'mpv': np.nan,
                    'sigma': np.nan,
                    'tau': np.nan,
                    'B': np.nan,
                    'x0': np.nan,
                    'C': np.nan,
                    'threshold': threshold,
                    'p_detect': np.nan,
                    'fit_error': str(e),
                })

        if shared_handles is not None:
            fig.legend(
                shared_handles,
                shared_labels,
                loc='upper right',
                bbox_to_anchor=(0.98, 0.98),
                ncol=4,
                fontsize=9,
                frameon=False,
                borderaxespad=0.4,
                columnspacing=0.8,
                handletextpad=0.4,
            )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

        filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
        base_filename = f'{filename_label}_{M1_or_M2}_sipm_pulseh_fits'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        summary_pkl_path = output_dir / f'{base_filename}_summary.pkl'
        summary_csv_path = output_dir / f'{base_filename}_summary.csv'

        plt.savefig(img_save_path)
        self.file_handler.save_pickle(fit_results_data, pkl_save_path)
        summary_df = pd.DataFrame(records).sort_values('channel')
        summary_df.to_pickle(summary_pkl_path)
        summary_df.to_csv(summary_csv_path, index=False)
        plt.close(fig)

        print(f"SiPM pulseH fits saved to {img_save_path}")
        print(f"SiPM pulseH fit data saved to {pkl_save_path}")
        print(f"SiPM pulseH fit summary saved to {summary_pkl_path}")

        return fit_results_data

    def _fit_and_plot_highlight_pe(self, highlight_events_df, mu1_values_run, mu1_errors_run, output_dir, file_label, M1_or_M2, fit_config):
        self.file_handler.ensure_dir(output_dir)

        bins = int(fit_config['bins'])
        hist_range = tuple(fit_config['hist_range'])
        sum_bins = int(fit_config.get('sum_bins', bins))
        sum_hist_range = tuple(fit_config.get('sum_hist_range', (hist_range[0], hist_range[1] * 12.0)))
        fit_half_width = float(fit_config['fit_window_half_width_pe'])
        min_fit_points = int(fit_config['min_fit_points'])

        edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        sum_edges = np.linspace(sum_hist_range[0], sum_hist_range[1], sum_bins + 1)
        sum_centers = 0.5 * (sum_edges[:-1] + sum_edges[1:])
        filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Highlight PMT P.E. Spectra (triggerBits=8) ({file_label}, {M1_or_M2})', fontsize=16)
        axes = axes.flatten()

        records = []
        hist_counts = {}
        fit_results_data = {}

        if highlight_events_df.empty:
            area_data = np.empty((0, len(config.PMT_CHANNELS)))
        else:
            area_data = np.array(highlight_events_df['area_array'].to_list())[:, config.PMT_CHANNELS]

        for i in range(12):
            ax = axes[i]
            mu1 = mu1_values_run[i] if i < len(mu1_values_run) else np.nan
            mu1_err = mu1_errors_run[i] if i < len(mu1_errors_run) else np.nan

            if area_data.size == 0 or (not np.isfinite(mu1)) or mu1 <= 0:
                counts = np.zeros(bins, dtype=float)
                hist_counts[i] = counts
                fit_results_data[i] = {
                    'counts': counts,
                    'edges': edges,
                    'popt': None,
                    'perr': None,
                    'peak_pe': np.nan,
                    'peak_pe_err': np.nan,
                    'peak_pe_err_fit': np.nan,
                    'peak_pe_err_mu1': np.nan,
                    'n_events': 0,
                }
                reason = 'No events' if area_data.size == 0 else 'Invalid $\\mu_1$'
                ax.text(0.5, 0.5, reason, transform=ax.transAxes, ha='center', va='center', color='red')
                ax.set_title(f'Channel {i}')
                ax.set_xlabel('P.E. (Area / $\\mu_1$)')
                ax.set_ylabel('Events')
                ax.set_xlim(hist_range)
                ax.grid(True, which='both', linestyle=':')
                records.append({'channel': i, 'peak_pe': np.nan, 'peak_pe_err': np.nan, 'n_events': 0})
                continue

            ch_pe = area_data[:, i].astype(float) / mu1
            ch_pe = ch_pe[np.isfinite(ch_pe)]
            counts, _ = np.histogram(ch_pe, bins=edges)
            hist_counts[i] = counts

            ax.step(edges, np.append(counts, counts[-1] if len(counts) > 0 else 0), where='post', alpha=0.8, label=f'N={len(ch_pe)}')

            peak_pe = np.nan
            peak_pe_err = np.nan
            peak_pe_err_fit = np.nan
            peak_pe_err_mu1 = np.nan
            popt_out = None
            perr_out = None

            if np.any(counts > 0):
                peak_idx = int(np.argmax(counts))
                peak_guess = float(centers[peak_idx])
                sigma_guess = max(float(np.std(ch_pe)) if ch_pe.size > 1 else 1.0, 0.2)
                amp_guess = max(float(np.max(counts) - np.min(counts)), 1.0)
                c_guess = float(np.min(counts))

                # Preferred fit region: FWHM window around peak (half-maximum crossings)
                y_half = 0.5 * float(counts[peak_idx])
                x_left = None
                x_right = None

                for idx_l in range(peak_idx, 0, -1):
                    y0 = float(counts[idx_l - 1])
                    y1 = float(counts[idx_l])
                    if (y0 <= y_half <= y1) or (y1 <= y_half <= y0):
                        x0 = float(centers[idx_l - 1])
                        x1 = float(centers[idx_l])
                        if abs(y1 - y0) > 1e-12:
                            frac = (y_half - y0) / (y1 - y0)
                            x_left = x0 + frac * (x1 - x0)
                        else:
                            x_left = x1
                        break

                for idx_r in range(peak_idx, len(counts) - 1):
                    y0 = float(counts[idx_r])
                    y1 = float(counts[idx_r + 1])
                    if (y0 >= y_half >= y1) or (y1 >= y_half >= y0):
                        x0 = float(centers[idx_r])
                        x1 = float(centers[idx_r + 1])
                        if abs(y1 - y0) > 1e-12:
                            frac = (y_half - y0) / (y1 - y0)
                            x_right = x0 + frac * (x1 - x0)
                        else:
                            x_right = x0
                        break

                if (x_left is not None) and (x_right is not None) and (x_right > x_left):
                    fit_lo = max(hist_range[0], x_left)
                    fit_hi = min(hist_range[1], x_right)
                else:
                    fit_lo = max(hist_range[0], peak_guess - fit_half_width)
                    fit_hi = min(hist_range[1], peak_guess + fit_half_width)

                fit_mask = (centers >= fit_lo) & (centers <= fit_hi) & (counts > 0)

                if np.count_nonzero(fit_mask) >= min_fit_points:
                    x_fit = centers[fit_mask]
                    y_fit = counts[fit_mask]
                    try:
                        popt, pcov = curve_fit(
                            self._gaussian,
                            x_fit,
                            y_fit,
                            p0=[amp_guess, peak_guess, sigma_guess, c_guess],
                            bounds=([0.0, hist_range[0], 1e-3, 0.0], [np.inf, hist_range[1], np.inf, np.inf]),
                            maxfev=30000
                        )
                        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)

                        peak_pe = float(popt[1])
                        peak_pe_err_fit = float(perr[1]) if len(perr) > 1 and np.isfinite(perr[1]) else np.nan
                        popt_out = popt
                        perr_out = perr

                        x_plot = np.linspace(fit_lo, fit_hi, 300)
                        ax.plot(x_plot, self._gaussian(x_plot, *popt), 'r-', linewidth=1.5,
                                label=f'Fit peak={peak_pe:.2f} p.e.')
                    except Exception:
                        peak_pe = peak_guess
                        peak_pe_err_fit = np.nan
                else:
                    peak_pe = peak_guess
                    peak_pe_err_fit = np.nan

            if np.isfinite(peak_pe) and np.isfinite(mu1_err) and np.isfinite(mu1) and mu1 > 0:
                peak_pe_err_mu1 = abs(peak_pe) * (mu1_err / mu1)
            if np.isfinite(peak_pe_err_fit) and np.isfinite(peak_pe_err_mu1):
                peak_pe_err = float(np.sqrt(peak_pe_err_fit**2 + peak_pe_err_mu1**2))
            elif np.isfinite(peak_pe_err_fit):
                peak_pe_err = float(peak_pe_err_fit)
            elif np.isfinite(peak_pe_err_mu1):
                peak_pe_err = float(peak_pe_err_mu1)
            else:
                peak_pe_err = np.nan

            fit_results_data[i] = {
                'counts': counts,
                'edges': edges,
                'popt': popt_out,
                'perr': perr_out,
                'peak_pe': peak_pe,
                'peak_pe_err': peak_pe_err,
                'peak_pe_err_fit': peak_pe_err_fit,
                'peak_pe_err_mu1': peak_pe_err_mu1,
                'n_events': int(ch_pe.size),
            }
            records.append({
                'channel': i,
                'peak_pe': peak_pe,
                'peak_pe_err': peak_pe_err,
                'peak_pe_err_fit': peak_pe_err_fit,
                'peak_pe_err_mu1': peak_pe_err_mu1,
                'n_events': int(ch_pe.size)
            })

            if np.isfinite(peak_pe):
                ax.text(0.98, 0.95, f'Peak={peak_pe:.2f} p.e.', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            ax.set_title(f'Channel {i}')
            ax.set_xlabel('P.E. (Area / $\\mu_1$)')
            ax.set_ylabel('Events')
            ax.set_xlim(hist_range)
            ax.grid(True, which='both', linestyle=':')
            ax.legend(loc='best', fontsize='small')

        # --- Summed highlight spectrum over PMT channels 0-11 ---
        sum_pe = np.array([], dtype=float)
        valid_mu1_channels = 0
        if area_data.size > 0:
            mu1_arr = np.asarray(mu1_values_run[:len(config.PMT_CHANNELS)], dtype=float)
            valid_mu1_mask = np.isfinite(mu1_arr) & (mu1_arr > 0)
            valid_mu1_channels = int(np.count_nonzero(valid_mu1_mask))
            if valid_mu1_channels > 0:
                pe_matrix = area_data[:, valid_mu1_mask].astype(float) / mu1_arr[valid_mu1_mask]
                pe_matrix = np.where(np.isfinite(pe_matrix), pe_matrix, np.nan)
                pe_matrix = np.clip(pe_matrix, 0.0, None)
                sum_pe = np.nansum(pe_matrix, axis=1)
                sum_pe = sum_pe[np.isfinite(sum_pe)]

        sum_counts, _ = np.histogram(sum_pe, bins=sum_edges)
        sum_peak_pe = np.nan
        sum_peak_pe_err = np.nan
        sum_sigma_pe = np.nan
        sum_sigma_pe_err = np.nan
        sum_popt = None
        sum_perr = None
        sum_fit_lo = np.nan
        sum_fit_hi = np.nan

        if np.any(sum_counts > 0):
            peak_idx = int(np.argmax(sum_counts))
            peak_guess = float(sum_centers[peak_idx])
            sigma_guess = max(float(np.std(sum_pe)) if sum_pe.size > 1 else 1.0, 0.2)
            amp_guess = max(float(np.max(sum_counts) - np.min(sum_counts)), 1.0)
            c_guess = float(np.min(sum_counts))

            y_half = 0.5 * float(sum_counts[peak_idx])
            x_left = None
            x_right = None

            for idx_l in range(peak_idx, 0, -1):
                y0 = float(sum_counts[idx_l - 1])
                y1 = float(sum_counts[idx_l])
                if (y0 <= y_half <= y1) or (y1 <= y_half <= y0):
                    x0 = float(sum_centers[idx_l - 1])
                    x1 = float(sum_centers[idx_l])
                    if abs(y1 - y0) > 1e-12:
                        frac = (y_half - y0) / (y1 - y0)
                        x_left = x0 + frac * (x1 - x0)
                    else:
                        x_left = x1
                    break

            for idx_r in range(peak_idx, len(sum_counts) - 1):
                y0 = float(sum_counts[idx_r])
                y1 = float(sum_counts[idx_r + 1])
                if (y0 >= y_half >= y1) or (y1 >= y_half >= y0):
                    x0 = float(sum_centers[idx_r])
                    x1 = float(sum_centers[idx_r + 1])
                    if abs(y1 - y0) > 1e-12:
                        frac = (y_half - y0) / (y1 - y0)
                        x_right = x0 + frac * (x1 - x0)
                    else:
                        x_right = x0
                    break

            if (x_left is not None) and (x_right is not None) and (x_right > x_left):
                sum_fit_lo = max(sum_hist_range[0], x_left)
                sum_fit_hi = min(sum_hist_range[1], x_right)
            else:
                sum_fit_lo = max(sum_hist_range[0], peak_guess - fit_half_width)
                sum_fit_hi = min(sum_hist_range[1], peak_guess + fit_half_width)

            fit_mask = (sum_centers >= sum_fit_lo) & (sum_centers <= sum_fit_hi) & (sum_counts > 0)
            if np.count_nonzero(fit_mask) >= min_fit_points:
                try:
                    popt, pcov = curve_fit(
                        self._gaussian,
                        sum_centers[fit_mask],
                        sum_counts[fit_mask],
                        p0=[amp_guess, peak_guess, sigma_guess, c_guess],
                        bounds=([0.0, sum_hist_range[0], 1e-3, 0.0], [np.inf, sum_hist_range[1], np.inf, np.inf]),
                        maxfev=30000
                    )
                    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
                    sum_popt = popt
                    sum_perr = perr
                    sum_peak_pe = float(popt[1])
                    sum_sigma_pe = float(popt[2])
                    sum_peak_pe_err = float(perr[1]) if len(perr) > 1 and np.isfinite(perr[1]) else np.nan
                    sum_sigma_pe_err = float(perr[2]) if len(perr) > 2 and np.isfinite(perr[2]) else np.nan
                except Exception:
                    sum_peak_pe = peak_guess
            else:
                sum_peak_pe = peak_guess

        sum_fig, sum_ax = plt.subplots(figsize=(10, 6))
        sum_ax.step(sum_edges, np.append(sum_counts, sum_counts[-1] if len(sum_counts) > 0 else 0),
                    where='post', alpha=0.9,
                    label=f'Sum over 12 PMTs (valid $\\mu_1$ channels={valid_mu1_channels}), N={int(sum_pe.size)}')
        if np.isfinite(sum_fit_lo) and np.isfinite(sum_fit_hi) and (sum_fit_hi > sum_fit_lo):
            sum_ax.axvspan(sum_fit_lo, sum_fit_hi, color='gray', alpha=0.18,
                           label=f'Fit window [{sum_fit_lo:.1f}, {sum_fit_hi:.1f}] p.e.')
        if sum_popt is not None:
            x_plot = np.linspace(sum_fit_lo, sum_fit_hi, 300)
            sum_ax.plot(x_plot, self._gaussian(x_plot, *sum_popt), 'r-', linewidth=1.6,
                        label=(
                            f'Gaussian fit: $\\mu$={sum_peak_pe:.2f}±{sum_peak_pe_err:.2f} p.e., '
                            f'$\\sigma$={sum_sigma_pe:.2f}±{sum_sigma_pe_err:.2f} p.e.'
                        ))
        elif np.isfinite(sum_peak_pe):
            sum_ax.axvline(sum_peak_pe, color='red', linestyle='--',
                           label=f'Peak estimate: {sum_peak_pe:.2f} p.e.')

        sum_ax.set_title(f'Highlight PMT P.E. Sum(Ch0-11) (triggerBits=8) ({file_label}, {M1_or_M2})')
        sum_ax.set_xlabel('Total P.E. (sum over 12 PMTs)')
        sum_ax.set_ylabel('Events')
        sum_ax.set_xlim(sum_hist_range)
        sum_ax.grid(True, which='both', linestyle=':')
        sum_ax.legend(loc='best', fontsize='small')
        sum_fig.tight_layout()

        sum_base_filename = f'{filename_label}_{M1_or_M2}_highlight_pe_sum12_fit'
        sum_img_save_path = output_dir / f'{sum_base_filename}.png'
        sum_pkl_save_path = output_dir / f'{sum_base_filename}.pkl'
        sum_summary_csv_path = output_dir / f'{sum_base_filename}_summary.csv'
        sum_summary_pkl_path = output_dir / f'{sum_base_filename}_summary.pkl'

        sum_fig.savefig(sum_img_save_path)
        plt.close(sum_fig)

        sum_pickle_data = {
            'counts': sum_counts,
            'edges': sum_edges,
            'popt': sum_popt,
            'perr': sum_perr,
            'peak_pe': sum_peak_pe,
            'peak_pe_err': sum_peak_pe_err,
            'sigma_pe': sum_sigma_pe,
            'sigma_pe_err': sum_sigma_pe_err,
            'fit_window': [float(sum_fit_lo), float(sum_fit_hi)],
            'n_events': int(sum_pe.size),
            'valid_mu1_channels': valid_mu1_channels,
        }
        self.file_handler.save_pickle(sum_pickle_data, sum_pkl_save_path)
        sum_summary_df = pd.DataFrame([
            {
                'peak_pe': sum_peak_pe,
                'peak_pe_err': sum_peak_pe_err,
                'sigma_pe': sum_sigma_pe,
                'sigma_pe_err': sum_sigma_pe_err,
                'fit_window_min': float(sum_fit_lo),
                'fit_window_max': float(sum_fit_hi),
                'n_events': int(sum_pe.size),
                'valid_mu1_channels': valid_mu1_channels,
            }
        ])
        sum_summary_df.to_csv(sum_summary_csv_path, index=False)
        sum_summary_df.to_pickle(sum_summary_pkl_path)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        base_filename = f'{filename_label}_{M1_or_M2}_highlight_pe_fits'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        summary_pkl_path = output_dir / f'{base_filename}_summary.pkl'
        summary_csv_path = output_dir / f'{base_filename}_summary.csv'

        plt.savefig(img_save_path)
        self.file_handler.save_pickle(fit_results_data, pkl_save_path)
        summary_df = pd.DataFrame(records).sort_values('channel')
        summary_df.to_pickle(summary_pkl_path)
        summary_df.to_csv(summary_csv_path, index=False)
        plt.close(fig)

        print(f"Highlight P.E. fits saved to {img_save_path}")
        print(f"Highlight P.E. fit data saved to {pkl_save_path}")
        print(f"Highlight P.E. fit summary saved to {summary_pkl_path}")
        print(f"Highlight 12-channel summed P.E. fit saved to {sum_img_save_path}")
        print(f"Highlight 12-channel summed P.E. fit data saved to {sum_pkl_save_path}")

        sum_hist_payload = {
            'counts': sum_counts,
            'edges': sum_edges,
            'n_events': int(sum_pe.size),
            'valid_mu1_channels': int(valid_mu1_channels),
        }
        return hist_counts, edges, summary_df, sum_hist_payload

    def _process_low_light_events(self, ll_events, ll_dir, run, M1_or_M2, low_light_fit_range):
        """Process low-light events and perform fitting."""
        low_light_area_data = np.array(ll_events['area_array'].to_list())[:, config.PMT_CHANNELS] if not ll_events.empty else np.array([])
        
        if low_light_area_data.size > 0:
            mu1_values, mu1_errors, fit_results_data = self._fit_and_plot_low_light(
                low_light_area_data, ll_dir, f'Run{run}', M1_or_M2, low_light_fit_range
            )
            
            # Extract histogram data for aggregation
            ll_hist_counts = {}
            ll_bin_edges = None
            for ch in range(12):
                if ch in fit_results_data:
                    ll_hist_counts[ch] = fit_results_data[ch]['counts']
                    if ll_bin_edges is None:
                        ll_bin_edges = fit_results_data[ch]['edges']
                else:
                    # If channel fit failed, create zero histogram
                    if ll_bin_edges is None:
                        ll_bin_edges = np.linspace(*low_light_fit_range, 201)
                    ll_hist_counts[ch] = np.zeros(len(ll_bin_edges) - 1)
            
            return mu1_values, mu1_errors, ll_hist_counts, ll_bin_edges
        else:
            print(f"No low-light events for run {run}. P.E. and multiplicity calculations will fail.")
            # Return empty histogram data
            ll_bin_edges = np.linspace(*low_light_fit_range, 201)
            ll_hist_counts = {ch: np.zeros(len(ll_bin_edges) - 1) for ch in range(12)}
            return np.full(12, np.nan), np.full(12, np.nan), ll_hist_counts, ll_bin_edges

    def _calculate_derived_quantities(self, df_all, mu1_values_run, multiplicity_spe):
        """Calculate derived quantities like multiplicity, time_std, and total_pe."""
        area_data_np = np.array(df_all['area_array'].to_list())[:, config.PMT_CHANNELS]
        times_data_np = np.array(df_all['peakPosition'].to_list())[:, config.PMT_CHANNELS]
        
        mu1_safe = np.where(np.isnan(mu1_values_run) | (mu1_values_run <= 0), np.inf, mu1_values_run)
        pe_per_channel = area_data_np / mu1_safe
        pe_per_channel = np.clip(pe_per_channel, 0.0, None)
        postmcut_mask = pe_per_channel > multiplicity_spe
        
        df_all['multiplicity'] = np.sum(postmcut_mask, axis=1)
        masked_times = np.where(postmcut_mask, times_data_np, np.nan)
        df_all['time_std'] = np.nanstd(masked_times, axis=1)
        df_all['total_pe'] = self.data_processor.calculate_total_pe(df_all, mu1_values_run)
        
        return df_all

    def _apply_cuts_and_generate_plots(self, df_all, run, hist_dir, cut_dir, delta_t_cut, pe_cut,
                                     bins, veto_bins, vetorange, multiplicity_cut, time_std_cut,
                                     logscale, M1_or_M2):
        """Apply event selection cuts and generate plots."""
        pe_min, pe_max = pe_cut
        
        # Apply cuts
        passing_cuts_mask = (
            (df_all['multiplicity'] >= multiplicity_cut) &
            (df_all['total_pe'] >= pe_min) & (df_all['total_pe'] <= pe_max) &
            (df_all['time_std'] < time_std_cut)
        )
        df_filtered = df_all[passing_cuts_mask & df_all['total_pe'].notna()]
        
        # Extract data for veto efficiency
        pe_trig2 = df_filtered.loc[(df_filtered['triggerBits'] == 2), 'total_pe']
        pe_trig2_or_34 = df_filtered.loc[
            (df_filtered['triggerBits'] == 2) | (df_filtered['triggerBits'] == 34), 'total_pe'
        ]
        
        # Plot comparisons
        if len(pe_trig2_or_34) + len(pe_trig2) > 0:
            pe_compare_data = np.concatenate([pe_trig2_or_34.values, pe_trig2.values])
            pe_compare_edges = self.plotter.hist_calc.bin_edges_from_spec(bins, pe_compare_data, pe_cut)
            
            self.plotter.plot_histogram(
                [pe_trig2_or_34.values, pe_trig2.values],
                ['Trig=2 or 34', 'Trig=2'],
                pe_compare_edges,
                hist_dir / f"{run}_{M1_or_M2}_total_pe_comparison.png",
                'Total PE Comparison', 'Total P.E.', M1_or_M2, logscale
            )

        # Plot veto efficiency
        veto_img_path = hist_dir / f"{run}_{M1_or_M2}_veto_efficiency.png"
        veto_pkl_path = hist_dir / f"{run}_{M1_or_M2}_veto_efficiency.pkl"
        veto_summary = self.plotter.plot_veto_efficiency(
            pe_trig2.to_numpy(), pe_trig2_or_34.to_numpy(),
            veto_bins, vetorange, pe_cut, veto_img_path, veto_pkl_path,
            f"Veto Efficiency Run {run}", M1_or_M2
        )

        # Process delta T analysis
        events = self.data_processor.compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)
        cut_payload = self._save_cut_histograms(events, delta_t_cut, pe_cut, bins, cut_dir,
                                               f"Run {run}", time_std_cut, M1_or_M2, logscale)

        if cut_payload is None:
            main_hist_payload = None
            michel_summary = {
                'success': False,
                'peak_location': np.nan,
                'peak_location_error': np.nan,
                'sigma': np.nan,
                'sigma_error': np.nan,
                'fwhm_min': np.nan,
                'fwhm_max': np.nan
            }
        else:
            main_hist_payload = cut_payload
            michel_summary = cut_payload.get('michel_fit', {
                'success': False,
                'peak_location': np.nan,
                'peak_location_error': np.nan,
                'sigma': np.nan,
                'sigma_error': np.nan,
                'fwhm_min': np.nan,
                'fwhm_max': np.nan
            })

        return main_hist_payload, pe_trig2, pe_trig2_or_34, veto_summary, michel_summary

    def _fit_michel_peak_fwhm(self, centers, counts, errors):
        """Fit Gaussian+constant in the FWHM region around the histogram peak."""
        centers = np.asarray(centers, dtype=float)
        counts = np.asarray(counts, dtype=float)
        errors = np.asarray(errors, dtype=float)

        fit_summary = {
            'success': False,
            'peak_location': np.nan,
            'peak_location_error': np.nan,
            'sigma': np.nan,
            'sigma_error': np.nan,
            'fwhm_min': np.nan,
            'fwhm_max': np.nan,
            'fit_x': np.array([]),
            'fit_y': np.array([]),
            'raw_peak': np.nan
        }

        if centers.size < 5 or counts.size != centers.size:
            return fit_summary

        peak_idx = int(np.argmax(counts))
        peak_count = float(counts[peak_idx])
        fit_summary['raw_peak'] = float(centers[peak_idx])
        if not np.isfinite(peak_count) or peak_count <= 0:
            return fit_summary

        half_max = 0.5 * peak_count
        above_half = counts >= half_max
        if not above_half[peak_idx]:
            return fit_summary

        left = peak_idx
        while left > 0 and above_half[left - 1]:
            left -= 1
        right = peak_idx
        while right < (len(counts) - 1) and above_half[right + 1]:
            right += 1

        x_fit = centers[left:right + 1]
        y_fit = counts[left:right + 1]
        e_fit = errors[left:right + 1]
        if x_fit.size < 4:
            return fit_summary

        def gauss_plus_const(x, amp, mu, sigma, c):
            sigma_safe = np.maximum(sigma, 1e-12)
            expo = np.clip(-0.5 * ((x - mu) / sigma_safe) ** 2, -700, 700)
            return amp * np.exp(expo) + c

        c0 = float(max(0.0, np.min(y_fit)))
        amp0 = float(max(np.max(y_fit) - c0, 1.0))
        mu0 = float(centers[peak_idx])
        fwhm_span = float(max(x_fit[-1] - x_fit[0], 1e-6))
        sigma0 = float(max(fwhm_span / 2.355, 1e-3))
        sigma_y = np.where(np.isfinite(e_fit) & (e_fit > 0), e_fit, 1.0)
        x_span_total = float(max(centers[-1] - centers[0], 1e-6))

        try:
            popt, pcov = curve_fit(
                gauss_plus_const,
                x_fit,
                y_fit,
                p0=[amp0, mu0, sigma0, c0],
                bounds=(
                    [0.0, x_fit[0], 1e-6, 0.0],
                    [np.inf, x_fit[-1], max(x_span_total, sigma0 * 10.0), np.inf]
                ),
                sigma=sigma_y,
                absolute_sigma=True,
                maxfev=100000
            )
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan, np.nan, np.nan, np.nan])
            mu_fit = float(popt[1])
            sigma_fit = float(np.abs(popt[2]))

            x_line = np.linspace(x_fit[0], x_fit[-1], 200)
            y_line = gauss_plus_const(x_line, *popt)

            fit_summary.update({
                'success': True,
                'peak_location': mu_fit,
                'peak_location_error': float(perr[1]),
                'sigma': sigma_fit,
                'sigma_error': float(perr[2]),
                'fwhm_min': float(x_fit[0]),
                'fwhm_max': float(x_fit[-1]),
                'fit_x': x_line,
                'fit_y': y_line
            })
        except Exception as e:
            print(f"Warning: Michel FWHM Gaussian fit failed. Error: {e}")

        return fit_summary

    def _fit_and_plot_low_light(self, area_data, output_dir, file_label, M1_or_M2, hist_range, hist_bins=200):
        """Plots and fits sum_area for channels 0-11 for low-light events."""
        if area_data.size == 0:
            print(f"No low-light data to process for {file_label}.")
            return np.full(12, np.nan), np.full(12, np.nan), {}

        plot_cfg = getattr(config, 'LOW_LIGHT_PLOT_CONFIG', {}) or {}
        figure_size = tuple(plot_cfg.get('figure_size', (22, 16)))
        dpi = int(plot_cfg.get('dpi', 300))
        suptitle_fontsize = int(plot_cfg.get('suptitle_fontsize', 20))
        channel_title_fontsize = int(plot_cfg.get('channel_title_fontsize', 15))
        axis_label_fontsize = int(plot_cfg.get('axis_label_fontsize', 13))
        tick_labelsize = int(plot_cfg.get('tick_labelsize', 12))
        legend_fontsize = int(plot_cfg.get('legend_fontsize', 11))
        annotation_fontsize = int(plot_cfg.get('annotation_fontsize', 11))

        def constrained_gaussians(x, a0, mu0, sig0, a1, mu1, sig1, a2, a3):
            sig2_sq = 2 * sig1**2 - sig0**2
            sig3_sq = 3 * sig1**2 - 2 * sig0**2
            if sig2_sq < 0 or sig3_sq < 0: 
                return np.inf
            pedestal = a0 * np.exp(-0.5 * ((x - mu0) / sig0)**2)
            spe = a1 * np.exp(-0.5 * ((x - mu1) / sig1)**2)
            dpe = a2 * np.exp(-0.5 * ((x - 2 * mu1) / np.sqrt(sig2_sq))**2)
            tpe = a3 * np.exp(-0.5 * ((x - 3 * mu1) / np.sqrt(sig3_sq))**2)
            return pedestal + spe + dpe + tpe

        fig, axes = plt.subplots(3, 4, figsize=figure_size)
        fig.suptitle(f'Low-Light Channel Area Fits ({file_label}, {M1_or_M2})', fontsize=suptitle_fontsize)
        axes = axes.flatten()
        
        mu1_values = np.full(12, np.nan)
        mu1_errors = np.full(12, np.nan)
        fit_results_data = {}

        for i in range(12):
            ax = axes[i]
            ch_data = area_data[:, i]
            counts, edges = np.histogram(ch_data, bins=hist_bins, range=hist_range)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.hist(ch_data, bins=edges, alpha=0.7, label=f'Ch {i} Data')

            p0 = [counts.max(), 0, 20, counts.max()/5, 100, 30, counts.max()/25, counts.max()/125]
            try:
                mask = counts > 0
                popt, pcov = curve_fit(constrained_gaussians, centers[mask], counts[mask], p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                mu1_values[i] = popt[4]
                if len(perr) > 4 and np.isfinite(perr[4]):
                    mu1_errors[i] = perr[4]
                fit_x = np.linspace(hist_range[0], hist_range[1], 500)
                ax.plot(fit_x, constrained_gaussians(fit_x, *popt), 'r-', linewidth=1.8, label='Fit')
                param_text = (f'$\\mu_1$: {popt[4]:.1f} ± {perr[4]:.1f}\n'
                              f'$\\sigma_1$: {popt[5]:.1f} ± {perr[5]:.1f}')
                ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=annotation_fontsize,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': popt, 'perr': perr}
            except (RuntimeError, ValueError):
                ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, color='red', ha='center', va='center', fontsize=annotation_fontsize)
                fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': None, 'perr': None}

            ax.set_title(f'Channel {i}', fontsize=channel_title_fontsize)
            ax.set_xlabel('Sum Area (ADC)', fontsize=axis_label_fontsize)
            ax.set_ylabel('Events', fontsize=axis_label_fontsize)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=tick_labelsize)
            ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.45)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.35)
            ax.minorticks_on()
            ax.legend(loc='lower left', fontsize=legend_fontsize)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self.file_handler.ensure_dir(output_dir)
        
        filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
        base_filename = f'{filename_label}_{M1_or_M2}_low_light_fits'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        
        plt.savefig(img_save_path, dpi=dpi)
        self.file_handler.save_pickle(fit_results_data, pkl_save_path)
        print(f"Low-light fits saved to {img_save_path}")
        print(f"Low-light fit data saved to {pkl_save_path}")
        plt.close()
        
        return mu1_values, mu1_errors, fit_results_data

    def _save_cut_histograms(self, events, delta_t_range, pe_range, bins,
                           save_dir, run_label, time_std_cut, M1_or_M2, logscale=True):
        """Apply sequential cuts and save errorbar histograms."""
        dt_min, dt_max = delta_t_range
        pe_min, pe_max = pe_range

        self.file_handler.ensure_dir(save_dir)
        sel = events.dropna(subset=['delta_t', 'total_pe']).copy()
        print(f"{run_label}: after NaN drop: {len(sel)} events")
        sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
        print(f"{run_label}: after Δt cut: {len(sel)} events")
        sel = sel[(sel['total_pe'] >= pe_min) & (sel['total_pe'] <= pe_max)]
        print(f"{run_label}: after total_pe cut: {len(sel)} events")

        sel = sel.dropna(subset=['time_std'])
        sel = sel[sel['time_std'] < time_std_cut]
        print(f"{run_label}: after time-std < {time_std_cut} ns cut: {len(sel)} events")

        self.plotter.plot_correlation_maps(sel, save_dir, run_label, M1_or_M2)

        if sel.empty:
            return None

        # Delta T Histogram
        dt_bins = self.plotter.hist_calc.make_dt_edges((dt_min, dt_max))
        dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
        dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
        dt_err = np.sqrt(dt_counts)
        
        dt_base_filename = f'delta_t_hist_{M1_or_M2}'
        self.file_handler.save_pickle(
            {'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err}, 
            save_dir / f'{dt_base_filename}.pkl'
        )
        
        plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
        plt.xlabel('Δt (ns)')
        dt_bin_width = float(np.median(np.diff(dt_edges))) if dt_edges.size > 1 else 0.0
        plt.ylabel(f'Counts per bin ({dt_bin_width:.1f} ns per bin)')
        plt.title(f'Δt Histogram ({M1_or_M2})')
        if logscale: 
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'{dt_base_filename}.png')
        plt.close()

        # Total PE Histogram
        pe_bins = self.plotter.hist_calc.bin_edges_from_spec(bins, sel['total_pe'].values, (pe_min, pe_max))
        pe_counts, pe_edges = np.histogram(sel['total_pe'], bins=pe_bins)
        pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
        peak_location = pe_centers[np.argmax(pe_counts)]
        peak = np.round(peak_location, 1)
        mean_pe = sel['total_pe'].mean()
        mean_pe_val = np.round(mean_pe, 1)
        pe_err = np.sqrt(pe_counts)
        michel_fit = self._fit_michel_peak_fwhm(pe_centers, pe_counts, pe_err)

        pe_base_filename = f'total_pe_hist_{M1_or_M2}'
        self.file_handler.save_pickle(
            {
                'hist': pe_counts,
                'centers': pe_centers,
                'errors': pe_err,
                'michel_fit': {
                    'success': bool(michel_fit.get('success', False)),
                    'peak_location': float(michel_fit.get('peak_location', np.nan)),
                    'peak_location_error': float(michel_fit.get('peak_location_error', np.nan)),
                    'sigma': float(michel_fit.get('sigma', np.nan)),
                    'sigma_error': float(michel_fit.get('sigma_error', np.nan)),
                    'fwhm_min': float(michel_fit.get('fwhm_min', np.nan)),
                    'fwhm_max': float(michel_fit.get('fwhm_max', np.nan)),
                    'raw_peak': float(michel_fit.get('raw_peak', np.nan))
                }
            }, 
            save_dir / f'{pe_base_filename}.pkl'
        )

        plot_label = f'{run_label}\nMean = {mean_pe_val} p.e.'
        plt.errorbar(pe_centers, pe_counts, yerr=pe_err, fmt='o', label=plot_label)
        
        plt.xlabel('Total Photoelectrons')
        pe_bin_width = float(np.median(np.diff(pe_edges))) if pe_edges.size > 1 else 0.0
        plt.ylabel(f'Counts per bin ({pe_bin_width:.1f} P.E. per bin)')
        plt.title(f'Total Photoelectron Histogram ({M1_or_M2})')
        plt.axvline(peak, color='gray', linestyle='--', label=f'Raw peak bin = {peak} p.e.')
        if michel_fit.get('success', False):
            fit_x = np.asarray(michel_fit['fit_x'], dtype=float)
            fit_y = np.asarray(michel_fit['fit_y'], dtype=float)
            plt.plot(
                fit_x,
                fit_y,
                color='red',
                linewidth=2.0,
                label=(
                    f"Michel fit in FWHM: $\\mu$={michel_fit['peak_location']:.2f}±{michel_fit['peak_location_error']:.2f}, "
                    f"$\\sigma$={michel_fit['sigma']:.2f}±{michel_fit['sigma_error']:.2f}"
                )
            )
            plt.axvline(
                michel_fit['peak_location'],
                color='red',
                linestyle=':',
                linewidth=1.6,
                label=f"Michel peak = {michel_fit['peak_location']:.2f} p.e."
            )
            plt.axvspan(
                michel_fit['fwhm_min'],
                michel_fit['fwhm_max'],
                color='red',
                alpha=0.12,
                label='FWHM fit window'
            )
        if logscale: 
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'{pe_base_filename}.png')
        plt.close()

        return {
            'delta_t': sel['delta_t'].values,
            'total_pe': sel['total_pe'].values,
            'multiplicity': sel['multiplicity'].values,
            'delta_t_hist_counts': dt_counts,
            'delta_t_hist_edges': dt_edges,
            'total_pe_hist_counts': pe_counts,
            'total_pe_hist_edges': pe_edges,
            'michel_fit': michel_fit
        }

def main():
    """Entry point for a single sub-job."""
    if len(sys.argv) < 4:
        print("Usage: python script.py <start_run> <end_run> <M1_or_M2> [output_dir] [step]")
        sys.exit(1)
        
    start_run = int(sys.argv[1])
    end_run = int(sys.argv[2])
    M1_or_M2 = sys.argv[3]
    top_output_dir = Path(sys.argv[4])
    step = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    if M1_or_M2 == 'M1':
        data_dir = Path(config.DATA_DIR_M1)
    elif M1_or_M2 == 'M2':
        data_dir = Path(config.DATA_DIR_M2)
    else:
        raise ValueError("M1_or_M2 must be 'M1' or 'M2'.")

    output_dir = top_output_dir / f"subjob_{start_run}-{end_run}"
    FileHandler.ensure_dir(output_dir)

    print("=== Configuration ===")
    print(f"Analysis type: {M1_or_M2}")
    print(f"Runs: {start_run} to {end_run}")
    print(f"Output Directory for this job: {output_dir}")
    print(f"Δt cut: {config.DELTA_T_CUT} ns")
    print(f"Photoelectron cut: {config.PE_CUT} P.E.")
    print(f"Time-std cut: < {config.TIME_STD_CUT} ns")
    print("======================")

    # Initialize processor and aggregated data
    processor = RunProcessor()
    aggregated = {
        'delta_t_hist': None,
        'delta_t_edges': None,
        'total_pe_hist': None,
        'total_pe_edges': None,
        'event61_hist': None,
        'event61_edges': None,
        'sipm_area_hists': None,
        'sipm_area_edges': None,
        'veto_histograms': None,
        'low_light_hists': [],
        'highlight_hists': [],
        'highlight_sum12_hists': [],
        'thin_veto_hists': [],
        'brn_channel_hists': None,
        'brn_delta_t_edges': None,
        'brn_area_edges': None,
    }
    ll_bin_edges_agg = None
    hl_bin_edges_agg = None
    hl_sum_edges_agg = None
    tv_bin_edges_agg = None
    total_subjob_timelength_ns = 0.0

    def accumulate_counts(existing_counts, existing_edges, new_counts, new_edges, label):
        new_counts = np.asarray(new_counts, dtype=float)
        new_edges = np.asarray(new_edges, dtype=float)
        if existing_counts is None:
            return new_counts.copy(), new_edges.copy()
        if existing_edges.shape != new_edges.shape or not np.allclose(existing_edges, new_edges):
            raise ValueError(f"Histogram edge mismatch while accumulating {label}")
        return existing_counts + new_counts, existing_edges

    # Process runs
    for run in range(start_run, end_run + 1, step):
        try:
            result_tuple = processor.process_run(
                run, data_dir, output_dir, config.DELTA_T_CUT, config.PE_CUT, 
                config.BINS, config.VETO_BINS, config.VETO_RANGE,
                config.MULTIPLICITY_SPE, config.MULTIPLICITY_CUT, config.TIME_STD_CUT, config.LOGSCALE_GENERAL,
                config.LOW_LIGHT_FIT_RANGE, {}, M1_or_M2
            )
            
            if result_tuple:
                result, run_timelength = result_tuple
                total_subjob_timelength_ns += run_timelength
                
                if result:
                    (main_hist_payload,
                     ll_hists, ll_edges,
                     hl_hists, hl_edges, hl_sum_payload,
                     event61_hist_payload,
                     sipm_hist_payload, veto_hist_payload,
                     tv_hists, tv_edges,
                     brn_hist_payload,
                     sipm_fit_data,
                     run_veto_summary) = result
                
                if main_hist_payload is not None:
                    aggregated['delta_t_hist'], aggregated['delta_t_edges'] = accumulate_counts(
                        aggregated['delta_t_hist'],
                        aggregated['delta_t_edges'],
                        main_hist_payload['delta_t_hist_counts'],
                        main_hist_payload['delta_t_hist_edges'],
                        'delta_t'
                    )
                    aggregated['total_pe_hist'], aggregated['total_pe_edges'] = accumulate_counts(
                        aggregated['total_pe_hist'],
                        aggregated['total_pe_edges'],
                        main_hist_payload['total_pe_hist_counts'],
                        main_hist_payload['total_pe_hist_edges'],
                        'total_pe'
                    )
                
                # Low-light histogram data
                aggregated['low_light_hists'].append(ll_hists)
                if ll_bin_edges_agg is None: 
                    ll_bin_edges_agg = ll_edges

                aggregated['highlight_hists'].append(hl_hists)
                if hl_bin_edges_agg is None:
                    hl_bin_edges_agg = hl_edges
                if hl_sum_payload is not None and np.asarray(hl_sum_payload.get('counts', [])).size > 0:
                    aggregated['highlight_sum12_hists'].append(np.asarray(hl_sum_payload.get('counts', []), dtype=float))
                    if hl_sum_edges_agg is None:
                        hl_sum_edges_agg = np.asarray(hl_sum_payload.get('edges', []), dtype=float)

                if event61_hist_payload is not None:
                    aggregated['event61_hist'], aggregated['event61_edges'] = accumulate_counts(
                        aggregated['event61_hist'],
                        aggregated['event61_edges'],
                        event61_hist_payload['counts'],
                        event61_hist_payload['edges'],
                        'event61'
                    )
                
                if sipm_hist_payload is not None:
                    if aggregated['sipm_area_hists'] is None:
                        aggregated['sipm_area_hists'] = {
                            ch: np.asarray(counts, dtype=float).copy()
                            for ch, counts in sipm_hist_payload['counts'].items()
                        }
                        aggregated['sipm_area_edges'] = np.asarray(sipm_hist_payload['edges'], dtype=float).copy()
                    else:
                        if aggregated['sipm_area_edges'].shape != np.asarray(sipm_hist_payload['edges']).shape or not np.allclose(aggregated['sipm_area_edges'], np.asarray(sipm_hist_payload['edges'], dtype=float)):
                            raise ValueError("SiPM histogram edge mismatch across runs")
                        for ch, counts in sipm_hist_payload['counts'].items():
                            aggregated['sipm_area_hists'][ch] += np.asarray(counts, dtype=float)

                if veto_hist_payload is not None:
                    if aggregated['veto_histograms'] is None:
                        aggregated['veto_histograms'] = {
                            'comparison_edges': np.asarray(veto_hist_payload['comparison_edges'], dtype=float).copy(),
                            'comparison_counts_2': np.asarray(veto_hist_payload['comparison_counts_2'], dtype=float).copy(),
                            'comparison_counts_2_or_34': np.asarray(veto_hist_payload['comparison_counts_2_or_34'], dtype=float).copy(),
                            'efficiency_edges': np.asarray(veto_hist_payload['efficiency_edges'], dtype=float).copy(),
                            'efficiency_counts_2': np.asarray(veto_hist_payload['efficiency_counts_2'], dtype=float).copy(),
                            'efficiency_counts_2_or_34': np.asarray(veto_hist_payload['efficiency_counts_2_or_34'], dtype=float).copy(),
                        }
                    else:
                        for edge_key in ('comparison_edges', 'efficiency_edges'):
                            if aggregated['veto_histograms'][edge_key].shape != np.asarray(veto_hist_payload[edge_key]).shape or not np.allclose(aggregated['veto_histograms'][edge_key], np.asarray(veto_hist_payload[edge_key], dtype=float)):
                                raise ValueError(f"Veto histogram edge mismatch for {edge_key}")
                        for count_key in ('comparison_counts_2', 'comparison_counts_2_or_34', 'efficiency_counts_2', 'efficiency_counts_2_or_34'):
                            aggregated['veto_histograms'][count_key] += np.asarray(veto_hist_payload[count_key], dtype=float)
                
                # Thin veto histogram data
                aggregated['thin_veto_hists'].append(tv_hists)
                if tv_bin_edges_agg is None:
                    tv_bin_edges_agg = tv_edges
                
                # BRN channel data
                if brn_hist_payload is not None:
                    brn_heatmap_shape = (
                        len(np.asarray(brn_hist_payload['delta_t_edges'], dtype=float)) - 1,
                        len(np.asarray(brn_hist_payload['area_edges'], dtype=float)) - 1,
                    )
                    if aggregated['brn_channel_hists'] is None:
                        aggregated['brn_channel_hists'] = {
                            ch: {
                                'delta_t': np.asarray(counts['delta_t'], dtype=float).copy(),
                                'area': np.asarray(counts['area'], dtype=float).copy(),
                                'delta_t_area': np.asarray(
                                    counts.get('delta_t_area', np.zeros(brn_heatmap_shape, dtype=float)),
                                    dtype=float,
                                ).copy(),
                            }
                            for ch, counts in brn_hist_payload['counts'].items()
                        }
                        aggregated['brn_delta_t_edges'] = np.asarray(brn_hist_payload['delta_t_edges'], dtype=float).copy()
                        aggregated['brn_area_edges'] = np.asarray(brn_hist_payload['area_edges'], dtype=float).copy()
                    else:
                        if aggregated['brn_delta_t_edges'].shape != np.asarray(brn_hist_payload['delta_t_edges']).shape or not np.allclose(aggregated['brn_delta_t_edges'], np.asarray(brn_hist_payload['delta_t_edges'], dtype=float)):
                            raise ValueError("BRN delta_t histogram edge mismatch across runs")
                        if aggregated['brn_area_edges'].shape != np.asarray(brn_hist_payload['area_edges']).shape or not np.allclose(aggregated['brn_area_edges'], np.asarray(brn_hist_payload['area_edges'], dtype=float)):
                            raise ValueError("BRN area histogram edge mismatch across runs")
                        for ch, counts in brn_hist_payload['counts'].items():
                            if ch not in aggregated['brn_channel_hists']:
                                aggregated['brn_channel_hists'][ch] = {
                                    'delta_t': np.asarray(counts['delta_t'], dtype=float).copy(),
                                    'area': np.asarray(counts['area'], dtype=float).copy(),
                                    'delta_t_area': np.asarray(
                                        counts.get('delta_t_area', np.zeros(brn_heatmap_shape, dtype=float)),
                                        dtype=float,
                                    ).copy(),
                                }
                            else:
                                aggregated['brn_channel_hists'][ch]['delta_t'] += np.asarray(counts['delta_t'], dtype=float)
                                aggregated['brn_channel_hists'][ch]['area'] += np.asarray(counts['area'], dtype=float)
                                aggregated['brn_channel_hists'][ch]['delta_t_area'] += np.asarray(
                                    counts.get('delta_t_area', np.zeros(brn_heatmap_shape, dtype=float)),
                                    dtype=float,
                                )
                    
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregated data
    print(f"Saving aggregated data for sub-job {start_run}-{end_run}...")
    try:
        with open(output_dir / 'subjob_format.json', 'w') as f:
            json.dump({'format_version': 2, 'artifact_mode': 'histogram-only'}, f, indent=4)

        # Save main histogram data
        if aggregated['delta_t_hist'] is not None and aggregated['delta_t_edges'] is not None:
            FileHandler.save_pickle(
                {'counts': aggregated['delta_t_hist'], 'edges': aggregated['delta_t_edges']},
                output_dir / 'aggregated_delta_t_hist.pkl'
            )
        if aggregated['total_pe_hist'] is not None and aggregated['total_pe_edges'] is not None:
            FileHandler.save_pickle(
                {'counts': aggregated['total_pe_hist'], 'edges': aggregated['total_pe_edges']},
                output_dir / 'aggregated_total_pe_hist.pkl'
            )
        if aggregated['event61_hist'] is not None and aggregated['event61_edges'] is not None:
            FileHandler.save_pickle(
                {'counts': aggregated['event61_hist'], 'edges': aggregated['event61_edges']},
                output_dir / 'aggregated_event61_hist.pkl'
            )

        # Save SiPM histogram data
        if aggregated['sipm_area_hists'] is not None and aggregated['sipm_area_edges'] is not None:
            FileHandler.save_pickle(
                {'counts': aggregated['sipm_area_hists'], 'edges': aggregated['sipm_area_edges']},
                output_dir / 'aggregated_sipm_area_hists.pkl'
            )

        # Save veto histogram data
        if aggregated['veto_histograms'] is not None:
            FileHandler.save_pickle(aggregated['veto_histograms'], output_dir / 'aggregated_veto_histograms.pkl')
        
        # Save low-light histogram data
        if aggregated['low_light_hists'] and ll_bin_edges_agg is not None:
            job_ll_master_counts = {ch: np.zeros_like(aggregated['low_light_hists'][0][ch]) for ch in range(12)}
            for run_hists in aggregated['low_light_hists']:
                for ch in range(12):
                    job_ll_master_counts[ch] += run_hists.get(ch, 0)
            ll_save_data = {'counts': job_ll_master_counts, 'edges': ll_bin_edges_agg}
            FileHandler.save_pickle(ll_save_data, output_dir / 'aggregated_low_light_hists.pkl')

        # Save highlight histogram data
        if aggregated['highlight_hists'] and hl_bin_edges_agg is not None:
            job_hl_master_counts = {ch: np.zeros_like(aggregated['highlight_hists'][0][ch]) for ch in range(12)}
            for run_hists in aggregated['highlight_hists']:
                for ch in range(12):
                    job_hl_master_counts[ch] += run_hists.get(ch, 0)
            hl_save_data = {'counts': job_hl_master_counts, 'edges': hl_bin_edges_agg}
            FileHandler.save_pickle(hl_save_data, output_dir / 'aggregated_highlight_hists.pkl')

        if aggregated['highlight_sum12_hists'] and hl_sum_edges_agg is not None:
            job_hl_sum12_counts = np.zeros_like(aggregated['highlight_sum12_hists'][0], dtype=float)
            for run_counts in aggregated['highlight_sum12_hists']:
                job_hl_sum12_counts += np.asarray(run_counts, dtype=float)
            hl_sum_save_data = {'counts': job_hl_sum12_counts, 'edges': hl_sum_edges_agg}
            FileHandler.save_pickle(hl_sum_save_data, output_dir / 'aggregated_highlight_sum12_hists.pkl')
        
        # Save thin veto histogram data
        if aggregated['thin_veto_hists'] and tv_bin_edges_agg is not None:
            hist_keys = aggregated['thin_veto_hists'][0].keys()
            job_tv_master_counts = {k: np.zeros_like(aggregated['thin_veto_hists'][0][k]) for k in hist_keys}
            for run_hists in aggregated['thin_veto_hists']:
                for k in hist_keys:
                    job_tv_master_counts[k] += run_hists.get(k, 0)
            tv_save_data = {'counts': job_tv_master_counts, 'edges': tv_bin_edges_agg}
            FileHandler.save_pickle(tv_save_data, output_dir / 'aggregated_thin_veto_hists.pkl')
        
        # Save BRN histogram data
        if config.PERFORM_BRN_ANALYSIS and aggregated['brn_channel_hists'] is not None:
            FileHandler.save_pickle(
                {
                    'counts': aggregated['brn_channel_hists'],
                    'delta_t_edges': aggregated['brn_delta_t_edges'],
                    'area_edges': aggregated['brn_area_edges'],
                },
                output_dir / 'aggregated_brn_channel_hists.pkl'
            )
        
        # Save sub-job time length
        subjob_time_data = {
            "timelength_ns": total_subjob_timelength_ns,
            "timelength_s": total_subjob_timelength_ns / 1e9,
            "timelength_min": total_subjob_timelength_ns / 1e9 / 60.0
        }
        with open(output_dir / "subjob_time_length.json", "w") as f:
            json.dump(subjob_time_data, f, indent=4)
        print(f"Sub-job time length saved to {output_dir / 'subjob_time_length.json'}")
            
        print("Successfully saved data for master aggregation.")

    except Exception as e:
        print(f"An error occurred while saving aggregated data: {e}")
        import traceback
        traceback.print_exc()

    print("--- Sub-job Analysis Complete ---")

if __name__ == "__main__":
    main()