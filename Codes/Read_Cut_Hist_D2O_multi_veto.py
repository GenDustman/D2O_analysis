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
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import uproot
import awkward as ak

# Import configuration or set defaults
try:
    import config
except ImportError:
    print("Warning: config module not found. Using default values.")
    class DefaultConfig:
        DATA_DIR_M1 = "/path/to/M1/data"
        DATA_DIR_M2 = "/path/to/M2/data"
        TIME_TICK_NS = 16
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
        BRN_HIST_CONFIG = {
            'area_range': (0, 5000),
            'area_bins': 100
        }
    config = DefaultConfig()

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
        
        area_data_np = np.array(df['area_array'].to_list())[:, :12]
        pe_per_channel = area_data_np / mu1_safe
        total_pe = np.sum(pe_per_channel, axis=1)
        
        return total_pe

    def compute_delta_t(self, df, muon_bits, veto_bits, mult_thresh):
        """Compute time differences Δt between veto events and the preceding muon event."""
        muon_mask = df['triggerBits'] >= muon_bits
        veto_mask = (df['triggerBits'] == veto_bits) & (df['multiplicity'] > mult_thresh)
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
    def compute_brn_data(df, pulseh_array, area_array, channels_to_analyze, brn_threshold):
        """
        Computes BRN delta_t and SiPM area data on a per-channel basis.
        - BRN delta_t: Time between SiPM event (trig 32/34) and previous beam-on (trig 0).
        - Data is stored only for channels that exceed the brn_threshold.
        """
        # Get times for beam-on (trig 0) and SiPM (trig 32/34) events
        beam_on_times = df.loc[df['triggerBits'] == 0, 'nsTime'].values
        sipm_events = df.loc[(df['triggerBits'] == 32) | (df['triggerBits'] == 34)].copy()

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
        sipm_indices = sipm_events.index
        sipm_pulseh_array = pulseh_array[sipm_indices]
        sipm_area_array = area_array[sipm_indices]
        all_brn_delta_t = sipm_events['brn_delta_t'].values

        # Initialize data structure
        channel_data = {ch: {'delta_t': [], 'area': []} for ch in channels_to_analyze}

        # Filter events by delta_t cut (apply to both delta_t and area data)
        dt_min, dt_max = config.BRN_DELTA_T_RANGE
        dt_cut_mask = (sipm_events['brn_delta_t'] >= dt_min) & (sipm_events['brn_delta_t'] <= dt_max)
        events_in_dt_range = sipm_events[dt_cut_mask]
        
        # Populate delta_t data (only for events in dt cut)
        if not events_in_dt_range.empty:
            filtered_pulseh_array_dt = pulseh_array[sipm_indices][dt_cut_mask]
            filtered_brn_delta_t = all_brn_delta_t[dt_cut_mask]
            
            for i in range(len(events_in_dt_range)):
                event_dt = filtered_brn_delta_t[i]
                if not np.isfinite(event_dt):
                    continue
                
                event_pulseh = filtered_pulseh_array_dt[i]
                for ch in channels_to_analyze:
                    if ch < len(event_pulseh) and event_pulseh[ch] > brn_threshold:
                        channel_data[ch]['delta_t'].append(event_dt)
        
            # Populate area data (using same filtered events)
            filtered_area_array = area_array[sipm_indices][dt_cut_mask]

            for i in range(len(events_in_dt_range)):
                event_pulseh = filtered_pulseh_array_dt[i]
                event_area = filtered_area_array[i]
                for ch in channels_to_analyze:
                    if ch < len(event_pulseh) and event_pulseh[ch] > brn_threshold:
                        channel_data[ch]['area'].append(event_area[ch])

        # Convert lists to numpy arrays
        for ch in channels_to_analyze:
            channel_data[ch]['delta_t'] = np.array(channel_data[ch]['delta_t'])
            channel_data[ch]['area'] = np.array(channel_data[ch]['area'])

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
            return

        pe_min, pe_max = pe_range
        bin_edges = np.linspace(pe_min, pe_max, bins + 1)
        
        counts_2, _ = np.histogram(trig2_pe, bins=bin_edges)
        counts_2_or_34, _ = np.histogram(trig2_or_34_pe, bins=bin_edges)
        
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Calculate Efficiency and Error
        efficiency = np.zeros_like(counts_2, dtype=float)
        error = np.zeros_like(counts_2, dtype=float)
        valid_mask = counts_2_or_34 > 0
        
        ratio = np.divide(counts_2[valid_mask], counts_2_or_34[valid_mask])
        efficiency[valid_mask] = 1 - ratio
        average_efficiency = np.mean(efficiency[valid_mask & (bin_centers >= vetorange[0]) & (bin_centers <= vetorange[1])])
        
        n = counts_2_or_34[valid_mask]
        p = ratio
        error[valid_mask] = np.sqrt(p * (1 - p) / n)

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
        plt.ylim(0, 1.1)
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

    def process_run(self, run, data_dir, output_dir, delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                    multiplicity_spe, multiplicity_cut, time_std_cut, logscale,
                    low_light_fit_range, simp_hist_config, M1_or_M2):
        """Process a single run: read data, perform calculations, and apply cuts."""
        print(f"--- Processing run {run} ---")
        
        # Determine input file path
        if M1_or_M2 == 'M1':
            infile = data_dir / (f"run{run}" + config.suffix_M1)
        elif M1_or_M2 == 'M2':
            infile = data_dir / (f"run{run}" + config.suffix_M2)
        else:
            raise ValueError("M1_or_M2 must be 'M1' or 'M2'")
            
        if not infile.exists():
            print(f"Missing file: {infile}")
            return None

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
        
        for dir_path in [hist_dir, cut_dir, ll_dir]:
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
        result = self._process_run_data(df_all, run, run_dir, hist_dir, cut_dir, ll_dir,
                                      delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                                      multiplicity_spe, multiplicity_cut, time_std_cut,
                                      logscale, low_light_fit_range, M1_or_M2)
        
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

    def _process_run_data(self, df_all, run, run_dir, hist_dir, cut_dir, ll_dir,
                         delta_t_cut, pe_cut, bins, veto_bins, vetorange,
                         multiplicity_spe, multiplicity_cut, time_std_cut,
                         logscale, low_light_fit_range, M1_or_M2):
        """Process the data for a single run."""
        
        # Plot trigger bits
        self.plotter.plot_histogram(
            [df_all['triggerBits'].to_numpy()], ['triggerBits'],
            np.arange(0, 37), hist_dir / f"{run}_{M1_or_M2}_triggerBits.png",
            f"Run {run} Trigger Bits", "Trigger Bits", M1_or_M2, logscale
        )
        
        # Process low-light events and calculate photoelectrons
        ll_events = df_all[df_all['triggerBits'] == 16]
        mu1_values_run, ll_hist_counts, ll_bin_edges = self._process_low_light_events(
            ll_events, ll_dir, run, M1_or_M2, low_light_fit_range
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
            df_all = df_all[time_interval_mask]
            
            print(f"Time interval cut: Kept {len(df_all)} / {original_event_count} events")
        # >>> END: NEW TIME INTERVAL CUT <<<
        
        # Save processed data
        df_all.to_pickle(run_dir / f"run{run}_{M1_or_M2}_data_with_pe.pkl")
        
        # Apply cuts and generate veto efficiency plots
        cut_results, pe_trig2, pe_trig2_or_34 = self._apply_cuts_and_generate_plots(
            df_all, run, hist_dir, cut_dir, delta_t_cut, pe_cut, bins, veto_bins,
            vetorange, multiplicity_cut, time_std_cut, logscale, M1_or_M2
        )
        
        # Extract SiPM events (triggerBits >= 32)
        sipm_events_df = df_all[df_all['triggerBits'] >= 32]
        
        # Initialize thin veto and BRN data
        tv_hist_counts = {}
        tv_bin_edges = {}
        brn_data = {}
        
        # Perform thin veto and BRN analysis if pulseH data is available
        if 'pulseH_array' in df_all.columns and (config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS):
            full_area_array = np.array(df_all['area_array'].to_list())
            pulseh_array = np.array(df_all['pulseH_array'].to_list())
            
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
                    config.BRN_SIPM_THRESHOLD_ADC
                )
                if brn_data:
                    BRNAnalyzer.plot_brn_histograms(
                        brn_data, hist_dir, f"Run {run}", M1_or_M2,
                        config.BRN_DELTA_T_RANGE, config.BRN_HIST_CONFIG
                    )
        else:
            if config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS:
                print(f"Warning: 'pulseH_array' not found for run {run}. Skipping thin veto and BRN analysis.")
            keys = ['muon_h', 'muon_a', 'no_co_h', 'no_co_a']
            tv_hist_counts, tv_bin_edges = self.plotter.hist_calc.calculate_histograms(
                {k: None for k in keys}, config.THIN_VETO_HIST_CONFIG
            )
        
        # Return all processed data
        if cut_results:
            dt_vals, pe_vals, mult_vals = cut_results
            return (dt_vals, pe_vals, mult_vals,
                   ll_hist_counts, ll_bin_edges,
                   sipm_events_df, pe_trig2, pe_trig2_or_34,
                   tv_hist_counts, tv_bin_edges,
                   brn_data)
        else:
            return (None, None, None,
                   ll_hist_counts, ll_bin_edges,
                   sipm_events_df, pd.Series(dtype=float), pd.Series(dtype=float),
                   tv_hist_counts, tv_bin_edges,
                   brn_data)

    def _process_low_light_events(self, ll_events, ll_dir, run, M1_or_M2, low_light_fit_range):
        """Process low-light events and perform fitting."""
        low_light_area_data = np.array(ll_events['area_array'].to_list())[:, :12] if not ll_events.empty else np.array([])
        
        if low_light_area_data.size > 0:
            mu1_values, fit_results_data = self._fit_and_plot_low_light(
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
            
            return mu1_values, ll_hist_counts, ll_bin_edges
        else:
            print(f"No low-light events for run {run}. P.E. and multiplicity calculations will fail.")
            # Return empty histogram data
            ll_bin_edges = np.linspace(*low_light_fit_range, 201)
            ll_hist_counts = {ch: np.zeros(len(ll_bin_edges) - 1) for ch in range(12)}
            return np.full(12, np.nan), ll_hist_counts, ll_bin_edges

    def _calculate_derived_quantities(self, df_all, mu1_values_run, multiplicity_spe):
        """Calculate derived quantities like multiplicity, time_std, and total_pe."""
        area_data_np = np.array(df_all['area_array'].to_list())[:, :12]
        times_data_np = np.array(df_all['peakPosition'].to_list())[:, :12]
        
        mu1_safe = np.where(np.isnan(mu1_values_run) | (mu1_values_run <= 0), np.inf, mu1_values_run)
        pe_per_channel = area_data_np / mu1_safe
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
        self.plotter.plot_veto_efficiency(
            pe_trig2.to_numpy(), pe_trig2_or_34.to_numpy(),
            veto_bins, vetorange, pe_cut, veto_img_path, veto_pkl_path,
            f"Veto Efficiency Run {run}", M1_or_M2
        )

        # Process delta T analysis
        events = self.data_processor.compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)
        cut_results = self._save_cut_histograms(events, delta_t_cut, pe_cut, bins, cut_dir,
                                               f"Run {run}", time_std_cut, M1_or_M2, logscale)
        
        return cut_results, pe_trig2, pe_trig2_or_34

    def _fit_and_plot_low_light(self, area_data, output_dir, file_label, M1_or_M2, hist_range, hist_bins=200):
        """Plots and fits sum_area for channels 0-11 for low-light events."""
        if area_data.size == 0:
            print(f"No low-light data to process for {file_label}.")
            return np.full(12, np.nan)

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

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Low-Light Channel Area Fits ({file_label}, {M1_or_M2})', fontsize=16)
        axes = axes.flatten()
        
        mu1_values = np.full(12, np.nan)
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
                fit_x = np.linspace(hist_range[0], hist_range[1], 500)
                ax.plot(fit_x, constrained_gaussians(fit_x, *popt), 'r-', label='Fit')
                param_text = (f'$\\mu_1$: {popt[4]:.1f} ± {perr[4]:.1f}\n'
                              f'$\\sigma_1$: {popt[5]:.1f} ± {perr[5]:.1f}')
                ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': popt, 'perr': perr}
            except (RuntimeError, ValueError):
                ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, color='red', ha='center', va='center')
                fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': None, 'perr': None}

            ax.set_title(f'Channel {i}')
            ax.set_xlabel('Sum Area (ADC)')
            ax.set_ylabel('Events')
            ax.grid(True, which='both', linestyle=':')
            ax.legend(loc='lower left', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self.file_handler.ensure_dir(output_dir)
        
        filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
        base_filename = f'{filename_label}_{M1_or_M2}_low_light_fits'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        
        plt.savefig(img_save_path)
        self.file_handler.save_pickle(fit_results_data, pkl_save_path)
        print(f"Low-light fits saved to {img_save_path}")
        print(f"Low-light fit data saved to {pkl_save_path}")
        plt.close()
        
        return mu1_values, fit_results_data

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
        plt.ylabel('Counts')
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

        pe_base_filename = f'total_pe_hist_{M1_or_M2}'
        self.file_handler.save_pickle(
            {'hist': pe_counts, 'centers': pe_centers, 'errors': pe_err}, 
            save_dir / f'{pe_base_filename}.pkl'
        )

        plot_label = f'{run_label}\nMean = {mean_pe_val} p.e.'
        plt.errorbar(pe_centers, pe_counts, yerr=pe_err, fmt='o', label=plot_label)
        
        plt.xlabel('Total Photoelectrons')
        plt.ylabel('Counts')
        plt.title(f'Total Photoelectron Histogram ({M1_or_M2})')
        plt.axvline(peak, color='red', linestyle='--', label=f'Peak = {peak} p.e.')
        if logscale: 
            plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f'{pe_base_filename}.png')
        plt.close()

        return sel['delta_t'].values, sel['total_pe'].values, sel['multiplicity'].values

def main():
    """Entry point for a single sub-job."""
    if len(sys.argv) != 5:
        print("Usage: python Read_Cut_Hist_D2O_multi_veto.py <start_run> <end_run> <M1_or_M2> <top_output_dir>")
        sys.exit(1)
        
    start_run = int(sys.argv[1])
    end_run = int(sys.argv[2])
    M1_or_M2 = sys.argv[3]
    top_output_dir = Path(sys.argv[4])
    
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
        'delta_t': [], 'total_pe': [], 'multiplicity': [],
        'sipm_events': [],
        'pe_trig2': [], 'pe_trig2_or_34': [],
        'low_light_hists': [],
        'thin_veto_hists': [],
        'brn_channel_data': []
    }
    ll_bin_edges_agg = None
    tv_bin_edges_agg = None
    total_subjob_timelength_ns = 0.0

    # Process runs
    for run in range(start_run, end_run + 1):
        try:
            result_tuple = processor.process_run(
                run, data_dir, output_dir, config.DELTA_T_CUT, config.PE_CUT, 
                config.BINS, config.VETO_BINS, config.VETO_RANGE,
                0.5, config.MULTIPLICITY_CUT, config.TIME_STD_CUT, True,
                config.LOW_LIGHT_FIT_RANGE, {}, M1_or_M2
            )
            
            if result_tuple:
                result, run_timelength = result_tuple
                total_subjob_timelength_ns += run_timelength
                
                if result:
                    (dt_vals, pe_vals, mult_vals,
                     ll_hists, ll_edges,
                     sipm_df, pe_2, pe_2_or_34,
                     tv_hists, tv_edges,
                     brn_data) = result
                
                if dt_vals is not None: 
                    aggregated['delta_t'].append(dt_vals)
                if pe_vals is not None: 
                    aggregated['total_pe'].append(pe_vals)
                if mult_vals is not None: 
                    aggregated['multiplicity'].append(mult_vals)
                
                # Low-light histogram data
                aggregated['low_light_hists'].append(ll_hists)
                if ll_bin_edges_agg is None: 
                    ll_bin_edges_agg = ll_edges
                
                if not sipm_df.empty: 
                    aggregated['sipm_events'].append(sipm_df)
                if not pe_2.empty: 
                    aggregated['pe_trig2'].append(pe_2)
                if not pe_2_or_34.empty: 
                    aggregated['pe_trig2_or_34'].append(pe_2_or_34)
                
                # Thin veto histogram data
                aggregated['thin_veto_hists'].append(tv_hists)
                if tv_bin_edges_agg is None:
                    tv_bin_edges_agg = tv_edges
                
                # BRN channel data
                if brn_data:
                    aggregated['brn_channel_data'].append(brn_data)
                    
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregated data
    print(f"Saving aggregated data for sub-job {start_run}-{end_run}...")
    try:
        # Save main arrays
        if aggregated['delta_t']:
            np.save(output_dir / 'aggregated_delta_t.npy', np.concatenate(aggregated['delta_t']))
            np.save(output_dir / 'aggregated_total_pe.npy', np.concatenate(aggregated['total_pe']))
            np.save(output_dir / 'aggregated_multiplicity.npy', np.concatenate(aggregated['multiplicity']))

        # Save SiPM area data
        if aggregated['sipm_events']:
            job_sipm_df = pd.concat(aggregated['sipm_events'], ignore_index=True)
            job_sipm_area_data = job_sipm_df['area_array']
            job_sipm_area_data.to_pickle(output_dir / 'aggregated_sipm_area_array.pkl')
        
        # Save veto efficiency data
        if aggregated['pe_trig2']:
            pd.concat(aggregated['pe_trig2'], ignore_index=True).to_pickle(output_dir / 'aggregated_pe_trig2.pkl')
        if aggregated['pe_trig2_or_34']:
            pd.concat(aggregated['pe_trig2_or_34'], ignore_index=True).to_pickle(output_dir / 'aggregated_pe_trig2_or_34.pkl')
        
        # Save low-light histogram data
        if aggregated['low_light_hists'] and ll_bin_edges_agg is not None:
            job_ll_master_counts = {ch: np.zeros_like(aggregated['low_light_hists'][0][ch]) for ch in range(12)}
            for run_hists in aggregated['low_light_hists']:
                for ch in range(12):
                    job_ll_master_counts[ch] += run_hists.get(ch, 0)
            ll_save_data = {'counts': job_ll_master_counts, 'edges': ll_bin_edges_agg}
            FileHandler.save_pickle(ll_save_data, output_dir / 'aggregated_low_light_hists.pkl')
        
        # Save thin veto histogram data
        if aggregated['thin_veto_hists'] and tv_bin_edges_agg is not None:
            hist_keys = aggregated['thin_veto_hists'][0].keys()
            job_tv_master_counts = {k: np.zeros_like(aggregated['thin_veto_hists'][0][k]) for k in hist_keys}
            for run_hists in aggregated['thin_veto_hists']:
                for k in hist_keys:
                    job_tv_master_counts[k] += run_hists.get(k, 0)
            tv_save_data = {'counts': job_tv_master_counts, 'edges': tv_bin_edges_agg}
            FileHandler.save_pickle(tv_save_data, output_dir / 'aggregated_thin_veto_hists.pkl')
        
        # Save BRN channel data
        if config.PERFORM_BRN_ANALYSIS and aggregated['brn_channel_data']:
            with open(output_dir / 'aggregated_brn_channel_data.pkl', 'wb') as f:
                pickle.dump(aggregated['brn_channel_data'], f)
        
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