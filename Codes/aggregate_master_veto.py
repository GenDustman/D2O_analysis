#!/usr/bin/env python3
"""
Master Aggregation Script for D2O Analysis (Refactored)

Reads all sub-job outputs and creates final, grand-aggregated results
using memory-efficient incremental aggregation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
import json
from datetime import datetime

# Import configuration
import config

# Import classes and functions from the refactored processing script
from Read_Cut_Hist_D2O_multi_veto import (
    FileHandler,
    HistogramCalculator,
    DataProcessor,
    Plotter
)

class DataAggregator:
    """Handles data aggregation operations."""
    
    def __init__(self):
        self.file_handler = FileHandler()
        self.hist_calc = HistogramCalculator()

    def incremental_concatenate(self, master_array, new_array_path, axis=0):
        """
        Loads a .npy file and appends it to a master array.
        If master_array is None, it creates it.
        This avoids holding a large list of arrays in memory.
        """
        if not new_array_path.exists():
            return master_array
        
        try:
            new_data = np.load(new_array_path)
        except (pickle.UnpicklingError, ValueError) as e:
            print(f"  Warning: Could not load {new_array_path.name}. File may be corrupt. Error: {e}")
            return master_array
            
        if new_data.size == 0:
            return master_array

        if master_array is None:
            return new_data
        else:
            return np.concatenate((master_array, new_data), axis=axis)

    def merge_channel_data_dicts(self, dict_list):
        """
        Merges a list of channel_data dictionaries from multiple runs/jobs.
        """
        if not dict_list:
            return {}
            
        try:
            all_channels = list(dict_list[0].keys())
        except (IndexError, AttributeError):
            print("Warning: BRN data list is empty or malformed.")
            return {}
            
        merged = {ch: {'delta_t': [], 'area': []} for ch in all_channels}
        
        for d in dict_list:
            if not isinstance(d, dict): 
                continue
            for ch, data in d.items():
                if ch in merged and isinstance(data, dict):
                    if data.get('delta_t', np.array([])).size > 0:
                        merged[ch]['delta_t'].append(data['delta_t'])
                    if data.get('area', np.array([])).size > 0:
                        merged[ch]['area'].append(data['area'])
        
        final_merged = {}
        for ch, data in merged.items():
            final_merged[ch] = {
                'delta_t': np.concatenate(data['delta_t']) if data['delta_t'] else np.array([]),
                'area': np.concatenate(data['area']) if data['area'] else np.array([])
            }
        return final_merged

class BinnedDataPlotter:
    """Handles plotting operations for pre-binned data."""
    
    def __init__(self):
        self.file_handler = FileHandler()
        self.hist_calc = HistogramCalculator()

    def plot_histogram_from_binned_data(self, hist_data_dict, bin_edges, img_path, title, xlabel, 
                                       M1_or_M2, logscale=True, figsize=(10, 6)):
        """
        Plots one or more overlapping histograms from pre-binned data.
        hist_data_dict = {'label1': counts_array1, 'label2': counts_array2}
        """
        plt.figure(figsize=figsize)
        outputs = {}
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        for label, counts in hist_data_dict.items():
            if counts.size > 0:
                total_n = np.sum(counts)
                plt.step(bin_edges, np.append(counts, counts[-1]), where='post', 
                        label=f"{label} (N={total_n:.0f})", alpha=0.7)
                outputs[label] = counts
            else:
                outputs[label] = np.zeros(len(bin_centers))

        plt.xlabel(xlabel)
        bin_width = float(np.median(np.diff(bin_edges))) if bin_edges.size > 1 else 0.0
        plt.ylabel('Events per Bin Width ({:.2f})'.format(bin_width))
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

        pkl_path = img_path.with_suffix('.pkl')
        pickle_data = {'centers': bin_centers, 'histograms': outputs}
        self.file_handler.save_pickle(pickle_data, pkl_path)
        plt.close()

    def plot_veto_efficiency_from_binned_data(self, counts_2, counts_2_or_34, bin_edges, vetorange, 
                                            img_path, pkl_path, title, M1_or_M2):
        """
        Calculates and plots veto efficiency from pre-binned counts.
        """
        if counts_2_or_34.sum() == 0:
            print(f"No events for veto efficiency calculation for {title}. Skipping.")
            return

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        efficiency = np.zeros_like(counts_2, dtype=float)
        error = np.zeros_like(counts_2, dtype=float)
        valid_mask = counts_2_or_34 > 0
        
        ratio = np.divide(counts_2[valid_mask], counts_2_or_34[valid_mask])
        efficiency[valid_mask] = 1 - ratio
        
        veto_range_mask = valid_mask & (bin_centers >= vetorange[0]) & (bin_centers <= vetorange[1])
        average_efficiency = np.mean(efficiency[veto_range_mask]) if np.any(veto_range_mask) else np.nan

        n = counts_2_or_34[valid_mask]
        p = ratio[valid_mask] 
        error[valid_mask] = np.sqrt(p * (1 - p) / n)
        err_average_efficiency = np.sqrt(np.mean(error[veto_range_mask]**2)) if np.any(veto_range_mask) else np.nan

        # --- Michel tail subtraction (fit exponential in specified range) ---
        def exp_model(x, a, b):
            return a * np.exp(b * x)

        michel_tail_counts = np.zeros_like(counts_2, dtype=float)
        michel_tail_variance = np.zeros_like(counts_2, dtype=float)
        michel_fit_params = None
        michel_fit_cov = None
        michel_fit_range = (vetorange[0] * 0.5, vetorange[1] * 0.5)

        fit_mask = (bin_centers >= michel_fit_range[0]) & (bin_centers <= michel_fit_range[1]) & (counts_2 > 0)
        if np.count_nonzero(fit_mask) >= 2:
            x_fit = bin_centers[fit_mask]
            y_fit = counts_2[fit_mask]
            try:
                a0 = max(y_fit.max(), 1.0)
                b0 = -1.0 / max((michel_fit_range[1] - michel_fit_range[0]), 1.0)
                popt, pcov = curve_fit(exp_model, x_fit, y_fit, p0=(a0, b0), bounds=([0.0, -np.inf], [np.inf, 0.0]))
                michel_fit_params = {'a': float(popt[0]), 'b': float(popt[1])}
                michel_fit_cov = pcov.tolist()
                michel_tail_counts = exp_model(bin_centers, *popt)
                michel_tail_counts = np.clip(michel_tail_counts, 0.0, None)

                # Propagate fit covariance to per-bin tail variance
                if pcov is not None and np.all(np.isfinite(pcov)):
                    exp_term = np.exp(popt[1] * bin_centers)
                    jac_a = exp_term
                    jac_b = popt[0] * bin_centers * exp_term
                    michel_tail_variance = (
                        jac_a * jac_a * pcov[0, 0]
                        + jac_b * jac_b * pcov[1, 1]
                        + 2.0 * jac_a * jac_b * pcov[0, 1]
                    )
                    michel_tail_variance = np.clip(michel_tail_variance, 0.0, None)
            except Exception as e:
                print(f"Warning: Michel tail fit failed for {title}. Error: {e}")

        # Subtracted efficiency
        counts_2_sub = counts_2 - michel_tail_counts
        counts_2_sub = np.clip(counts_2_sub, 0.0, None)
        counts_2_or_34_sub = counts_2_or_34 - michel_tail_counts
        counts_2_or_34_sub = np.clip(counts_2_or_34_sub, 0.0, None)

        label_raw = r'Raw: $\varepsilon_{\mathrm{veto}}$'
        label_sub = r'Michel-sub: $\varepsilon_{\mathrm{veto}}$'

        efficiency_sub = np.zeros_like(counts_2, dtype=float)
        error_sub = np.zeros_like(counts_2, dtype=float)
        valid_mask_sub = counts_2_or_34_sub > 0
        veto_range_mask_sub = valid_mask_sub & (bin_centers >= vetorange[0]) & (bin_centers <= vetorange[1])
        ratio_sub = np.divide(counts_2_sub[valid_mask_sub], counts_2_or_34_sub[valid_mask_sub])
        efficiency_sub[valid_mask_sub] = 1 - ratio_sub

        # Error propagation for r=(N-S)/(D-S): include Poisson N,D and shared Michel-tail S uncertainty
        denom_sub = counts_2_or_34_sub[valid_mask_sub]
        var_num = counts_2[valid_mask_sub]
        var_den = counts_2_or_34[valid_mask_sub]
        var_tail = michel_tail_variance[valid_mask_sub]
        d_ratio_d_num = 1.0 / np.maximum(denom_sub, 1e-12)
        d_ratio_d_den = -ratio_sub / np.maximum(denom_sub, 1e-12)
        d_ratio_d_tail = (counts_2[valid_mask_sub] - counts_2_or_34[valid_mask_sub]) / np.maximum(denom_sub, 1e-12) ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            var_ratio = (
                (d_ratio_d_num ** 2) * var_num
                + (d_ratio_d_den ** 2) * var_den
                + (d_ratio_d_tail ** 2) * var_tail
            )
        var_ratio = np.clip(var_ratio, 0.0, None)
        error_sub[valid_mask_sub] = np.sqrt(var_ratio)

        plt.figure(figsize=(10, 6))
        plt.errorbar(bin_centers[valid_mask], efficiency[valid_mask], yerr=error[valid_mask],
                     fmt='o', capsize=3, label=label_raw, color='navy', markersize=5)
        plt.errorbar(bin_centers[valid_mask_sub], efficiency_sub[valid_mask_sub], yerr=error_sub[valid_mask_sub],
                 fmt='s', capsize=3, label=label_sub, color='darkorange', markersize=4)
        
        if not np.isnan(average_efficiency):
            plt.axhline(average_efficiency, color='red', linestyle='--',
                        label=f'Average Efficiency = {100*average_efficiency:.4f}% ± {100*err_average_efficiency:.4f}%')

        average_efficiency_sub = np.mean(efficiency_sub[veto_range_mask_sub]) if np.any(veto_range_mask_sub) else np.nan
        err_average_efficiency_sub = np.sqrt(np.mean(error_sub[veto_range_mask_sub]**2)) if np.any(veto_range_mask_sub) else np.nan
        if not np.isnan(average_efficiency_sub):
            plt.axhline(average_efficiency_sub, color='orange', linestyle=':',
                        label=f'Average Efficiency (Michel-sub) = {100*average_efficiency_sub:.4f}% ± {100*err_average_efficiency_sub:.4f}%')
        
        plt.xlabel('Total Photoelectrons (P.E.)')
        plt.ylabel('Veto Efficiency')
        plt.title(f"{title} ({M1_or_M2})")
        plt.xlim(vetorange)
        plt.ylim(0.995, 1.002)
        plt.grid(which='major', linestyle='-', linewidth=0.7)
        plt.grid(which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        plt.tight_layout()
        plt.legend()
        self.file_handler.ensure_dir(img_path.parent)
        plt.savefig(img_path)
        plt.close()

        pickle_data = {
            'centers': bin_centers, 'efficiency': efficiency, 'error': error,
            'efficiency_subtracted': efficiency_sub, 'error_subtracted': error_sub,
            'counts_2': counts_2, 'counts_2_or_34': counts_2_or_34,
            'michel_tail_counts': michel_tail_counts,
            'michel_fit_params': michel_fit_params,
            'michel_fit_cov': michel_fit_cov,
            'michel_fit_range': michel_fit_range,
            'michel_tail_variance': michel_tail_variance
        }
        self.file_handler.save_pickle(pickle_data, pkl_path)
        print(f"Veto efficiency plot saved to {img_path}")
        print(f"Veto efficiency data saved to {pkl_path}")

    def plot_sipm_histograms_from_binned_data(self, hist_data, bin_edges, output_dir, label, M1_or_M2, hist_config):
        """
        Plots SiPM histograms from pre-binned data (dict of counts).
        """
        self.file_handler.ensure_dir(output_dir)
        filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'SiPM Channel Area (triggerBits>=32) - {label} ({M1_or_M2})', fontsize=16)
        axes = axes.flatten()
        
        sipm_hist_data = {}
        sipm_channels = config.SIPM_CHANNELS
        
        for i, ch in enumerate(sipm_channels):
            ax = axes[i]
            if ch in hist_data and hist_data[ch] is not None:
                counts = hist_data[ch]
                total_events = np.sum(counts)
                
                ax.step(bin_edges, np.append(counts, counts[-1]), where='post', color='darkcyan', 
                       label=f"N = {total_events:.0f}")
                
                sipm_hist_data[ch] = {'counts': counts, 'edges': bin_edges}
                ax.set_title(f'SiPM Channel {ch}')
                ax.set_xlabel('Area (ADC)')
                ax.set_ylabel('Events')
                ax.grid(True, which='both', linestyle=':')
                ax.set_yscale('log')
                ax.set_xlim(hist_config['hist_range'])
                if total_events > 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, f'Channel {ch}\nNo Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
        
        for i in range(len(sipm_channels), len(axes)):
            axes[i].set_axis_off()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        base_filename = f'{filename_label}_{M1_or_M2}_sipm_area_histograms'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        
        plt.savefig(img_save_path)
        self.file_handler.save_pickle(sipm_hist_data, pkl_save_path)
        print(f"SiPM histograms saved to {img_save_path}")
        print(f"SiPM histogram data saved to {pkl_save_path}")
        plt.close(fig)

    def plot_normalized_histogram_comparison_from_binned_data(self, counts1, label1, counts2, label2, 
                                                            bin_edges, img_path, title, xlabel, 
                                                            M1_or_M2, figsize=(10, 6)):
        """
        Plots two datasets as overlapping, normalized histograms from pre-binned counts.
        Uses a log scale on the y-axis.
        """
        plt.figure(figsize=figsize)
        outputs = {}
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        total1 = counts1.sum()
        if total1 > 0:
            density1 = counts1 / (total1 * np.diff(bin_edges))
            plt.step(bin_edges, np.append(density1, density1[-1]), where='post', 
                     label=f"{label1} (N={total1:.0f})", alpha=0.7)
            outputs[label1] = counts1

        total2 = counts2.sum()
        if total2 > 0:
            density2 = counts2 / (total2 * np.diff(bin_edges))
            plt.step(bin_edges, np.append(density2, density2[-1]), where='post', linewidth=2, 
                     label=f"{label2} (N={total2:.0f})", alpha=0.7)
            outputs[label2] = counts2

        plt.xlabel(xlabel)
        bin_width = float(np.median(np.diff(bin_edges))) if bin_edges.size > 1 else 0.0
        plt.ylabel(f'Normalized Events / Bin Width ({bin_width:.2f})')
        plt.title(f"{title} ({M1_or_M2})")
        plt.yscale('log')
        plt.legend()
        plt.minorticks_on()
        plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
        plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
        plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
        plt.tight_layout()
        plt.savefig(img_path)
        
        pkl_path = img_path.with_suffix('.pkl')
        if outputs:
            pickle_data = {'centers': bin_centers, 'histograms': outputs, 'edges': bin_edges}
            self.file_handler.save_pickle(pickle_data, pkl_path)
        
        plt.close()

    def fit_and_plot_low_light_from_binned_data(self, hist_data, bin_edges, output_dir, file_label, 
                                               M1_or_M2, hist_range):
        """
        Plots and fits sum_area for channels 0-11 from pre-binned data.
        """
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
        
        fit_results_data = {}
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for i in range(12):  # Channels 0-11
            ax = axes[i]
            counts = hist_data.get(i, np.zeros(len(centers)))

            ax.step(bin_edges, np.append(counts, counts[-1]), where='post', label=f'Ch {i} Data', alpha=0.7)

            p0 = [counts.max(), 0, 20, counts.max()/5, 100, 30, counts.max()/25, counts.max()/125]
            try:
                mask = counts > 0
                if not np.any(mask):
                    raise RuntimeError("No data to fit")
                    
                popt, pcov = curve_fit(constrained_gaussians, centers[mask], counts[mask], p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                fit_x = np.linspace(hist_range[0], hist_range[1], 500)
                ax.plot(fit_x, constrained_gaussians(fit_x, *popt), 'r-', label='Fit')
                param_text = (f'$\\mu_1$: {popt[4]:.1f} $\\pm$ {perr[4]:.1f}\n'
                              f'$\\sigma_1$: {popt[5]:.1f} $\\pm$ {perr[5]:.1f}')
                ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                fit_results_data[i] = {'counts': counts, 'edges': bin_edges, 'popt': popt, 'perr': perr}
            except (RuntimeError, ValueError):
                ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, color='red', ha='center', va='center')
                fit_results_data[i] = {'counts': counts, 'edges': bin_edges, 'popt': None, 'perr': None}

            ax.set_title(f'Channel {i}')
            ax.set_xlabel('Sum Area (ADC)')
            ax.set_ylabel('Events')
            ax.set_xlim(hist_range)
            ax.grid(True, which='both', linestyle=':')
            ax.legend(loc='best', fontsize='small')

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
        plt.close(fig)

    def fit_and_plot_highlight_from_binned_data(self, hist_data, bin_edges, output_dir, file_label,
                                                M1_or_M2, hist_range):
        """Plots and fits highlight P.E. spectra for PMT channels 0-11 from pre-binned data."""
        def gaussian(x, amp, mu, sigma, c):
            sigma = np.maximum(sigma, 1e-6)
            return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Highlight PMT P.E. Fits ({file_label}, {M1_or_M2})', fontsize=16)
        axes = axes.flatten()

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        fit_results_data = {}
        records = []

        for i in range(12):
            ax = axes[i]
            counts = np.asarray(hist_data.get(i, np.zeros(len(centers))), dtype=float)
            ax.step(bin_edges, np.append(counts, counts[-1] if len(counts) > 0 else 0), where='post', alpha=0.8, label=f'Ch {i}')

            peak_pe = np.nan
            peak_pe_err = np.nan
            popt_out = None
            perr_out = None

            if np.any(counts > 0):
                peak_idx = int(np.argmax(counts))
                peak_guess = float(centers[peak_idx])
                amp_guess = max(float(np.max(counts) - np.min(counts)), 1.0)
                total_w = float(np.sum(counts))
                if total_w > 1:
                    mean_w = float(np.sum(centers * counts) / total_w)
                    var_w = float(np.sum(counts * (centers - mean_w) ** 2) / total_w)
                    sigma_guess = max(np.sqrt(max(var_w, 1e-6)), 0.2)
                else:
                    sigma_guess = 1.0
                c_guess = float(np.min(counts))

                fit_half_width = float(getattr(config, 'HIGHLIGHT_FIT_CONFIG', {}).get('fit_window_half_width_pe', 12.0))
                min_fit_points = int(getattr(config, 'HIGHLIGHT_FIT_CONFIG', {}).get('min_fit_points', 6))

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
                    try:
                        popt, pcov = curve_fit(
                            gaussian,
                            centers[fit_mask],
                            counts[fit_mask],
                            p0=[amp_guess, peak_guess, sigma_guess, c_guess],
                            bounds=([0.0, hist_range[0], 1e-3, 0.0], [np.inf, hist_range[1], np.inf, np.inf]),
                            maxfev=30000
                        )
                        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(len(popt), np.nan)
                        peak_pe = float(popt[1])
                        peak_pe_err = float(perr[1]) if len(perr) > 1 and np.isfinite(perr[1]) else np.nan
                        popt_out = popt
                        perr_out = perr
                        x_plot = np.linspace(fit_lo, fit_hi, 300)
                        ax.plot(x_plot, gaussian(x_plot, *popt), 'r-', linewidth=1.5, label=f'Fit peak={peak_pe:.2f} p.e.')
                    except Exception:
                        peak_pe = peak_guess
                else:
                    peak_pe = peak_guess

            fit_results_data[i] = {
                'counts': counts,
                'edges': bin_edges,
                'popt': popt_out,
                'perr': perr_out,
                'peak_pe': peak_pe,
                'peak_pe_err': peak_pe_err,
            }
            records.append({'channel': i, 'peak_pe': peak_pe, 'peak_pe_err': peak_pe_err})

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

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        self.file_handler.ensure_dir(output_dir)

        filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
        base_filename = f'{filename_label}_{M1_or_M2}_highlight_pe_fits_master'
        img_save_path = output_dir / f'{base_filename}.png'
        pkl_save_path = output_dir / f'{base_filename}.pkl'
        summary_pkl_path = output_dir / f'{base_filename}_summary.pkl'
        summary_csv_path = output_dir / f'{base_filename}_summary.csv'

        plt.savefig(img_save_path)
        self.file_handler.save_pickle(fit_results_data, pkl_save_path)
        summary_df = pd.DataFrame(records).sort_values('channel')
        summary_df.to_pickle(summary_pkl_path)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Highlight fits saved to {img_save_path}")
        print(f"Highlight fit data saved to {pkl_save_path}")
        plt.close(fig)

# Simplified aggregate_plots function for compatibility
def aggregate_plots(aggregated_data, delta_t_cut, pe_cut, bins, tau_fit_window,
                   output_dir, m1_or_m2, agg_label, logscale_dt, logscale_pe, do_tau_fit):
    """Simplified aggregate plots function."""
    print(f"Generating aggregate plots for {agg_label}")
    
    if not aggregated_data['delta_t'] or len(aggregated_data['delta_t'][0]) == 0:
        print("No delta_t data to plot")
        return
        
    # Create basic plots - this is a placeholder implementation
    # You would need to implement the full plotting logic here
    plotter = Plotter()
    hist_calc = HistogramCalculator()
    
    dt_data = aggregated_data['delta_t'][0]
    pe_data = aggregated_data['total_pe'][0]
    
    # Delta T histogram
    dt_edges = hist_calc.make_dt_edges(delta_t_cut)
    plotter.plot_histogram(
        [dt_data], [agg_label], dt_edges,
        output_dir / f"aggregated_delta_t_{m1_or_m2}.png",
        f"Aggregated Delta T {agg_label}", "Delta T (ns)", m1_or_m2, logscale_dt
    )
    if do_tau_fit:
        fit_window = config.TAU_FIT_WINDOW
    
    # PE histogram
    pe_edges = hist_calc.bin_edges_from_spec(bins, pe_data, pe_cut)
    pe_counts, _ = np.histogram(pe_data, bins=pe_edges)
    pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
    pe_err = np.sqrt(pe_counts)

    # Exponential fit for aggregated total PE
    def exp_model(x, a, b):
        return a * np.exp(b * x)

    fit_range = (config.VETO_RANGE[0] * 0.5, config.VETO_RANGE[1] * 0.5)
    fit_mask = (pe_counts > 0) & (pe_centers >= fit_range[0]) & (pe_centers <= fit_range[1])
    fit_params = None
    fit_curve = None
    if np.count_nonzero(fit_mask) >= 2:
        x_fit = pe_centers[fit_mask]
        y_fit = pe_counts[fit_mask]
        try:
            a0 = max(y_fit.max(), 1.0)
            b0 = -1.0 / max((pe_centers.max() - pe_centers.min()), 1.0)
            popt, _ = curve_fit(
                exp_model,
                x_fit,
                y_fit,
                p0=(a0, b0),
                bounds=([0.0, -np.inf], [np.inf, 0.0])
            )
            fit_params = {'a': float(popt[0]), 'b': float(popt[1])}
            pe_centers_fit = pe_centers[fit_mask]
            fit_curve = exp_model(pe_centers_fit, *popt)
            fit_curve = np.clip(fit_curve, 0.0, None)
        except Exception as e:
            print(f"Warning: Exponential fit failed for aggregated total PE. Error: {e}")

    plt.figure(figsize=(10, 6))
    plt.errorbar(pe_centers, pe_counts, yerr=pe_err, fmt='o', label=agg_label, markersize=5, zorder=2)
    if fit_curve is not None:
        plt.plot(pe_centers_fit, fit_curve, 'r-', label='Exp fit, a={:.2f}, b={:.2f}'.format(fit_params['a'], fit_params['b']), zorder=3)
    # Shade veto-efficiency fit window
    plt.axvspan(fit_range[0], fit_range[1], color='gray', alpha=0.3, label='Fit Window')
    plt.xlabel('Total P.E.')
    bin_width_pe = float(np.median(np.diff(pe_edges))) if pe_edges.size > 1 else 0.0
    plt.ylabel(f'Counts per bin ({bin_width_pe:.1f} P.E. per bin)')
    plt.title(f"Aggregated Total PE {agg_label} ({m1_or_m2})")
    if logscale_pe:
        plt.yscale('log')
        plt.ylim(bottom=0.8)
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    pe_img_path = output_dir / f"aggregated_total_pe_{m1_or_m2}.png"
    plt.savefig(pe_img_path)
    plt.close()

    pe_pkl_path = pe_img_path.with_suffix('.pkl')
    pickle_data = {
        'centers': pe_centers,
        'histograms': {agg_label: pe_counts},
        'errors': {agg_label: pe_err},
        'fit_params': fit_params
    }
    plotter.file_handler.save_pickle(pickle_data, pe_pkl_path)

class MasterAggregator:
    """Main aggregator class for combining all sub-job results."""
    
    def __init__(self, top_dir_path):
        self.top_dir = Path(top_dir_path)
        if not self.top_dir.is_dir():
            raise FileNotFoundError(f"Error: Directory not found at {self.top_dir}")

        # Initialize helper classes
        self.file_handler = FileHandler()
        self.data_aggregator = DataAggregator()
        self.binned_plotter = BinnedDataPlotter()
        self.plotter = Plotter()

        # Setup output directory
        self.master_output_dir = self.top_dir / "MASTER_RESULTS"
        self.file_handler.ensure_dir(self.master_output_dir)
        print(f"Master output will be saved to: {self.master_output_dir}")

        # Find sub-job directories
        self.subjob_dirs = sorted(list(self.top_dir.glob("subjob_*")))
        if not self.subjob_dirs:
            raise FileNotFoundError("Error: No 'subjob_*' directories found. Did the jobs run correctly?")
        
        print(f"Found {len(self.subjob_dirs)} sub-job directories to aggregate.")
        
        # Initialize labels from directory name
        self._initialize_labels()
        
        # Initialize data containers
        self._initialize_data_containers()

    def _initialize_labels(self):
        """Initialize labels from directory name."""
        dir_name_parts = self.top_dir.name.split('_')
        self.run_range_str = dir_name_parts[1]
        self.m1_or_m2 = dir_name_parts[2]
        self.agg_label = f"Master Runs {self.run_range_str}"
        self.filename_label = self.agg_label.replace(" ", "_").replace("-", "_")

    def _initialize_data_containers(self):
        """Initialize all data container variables."""
        # Main arrays
        self.master_dt, self.master_pe, self.master_mult = None, None, None
        
        # BRN data
        self.all_brn_data = []

        # SiPM data
        self.sipm_channels = config.SIPM_CHANNELS
        self.sipm_bin_edges = np.linspace(*config.SIPM_HIST_CONFIG['hist_range'], 
                                         config.SIPM_HIST_CONFIG['hist_bins'] + 1)
        self.master_sipm_hist_counts = {ch: np.zeros(config.SIPM_HIST_CONFIG['hist_bins']) 
                                       for ch in self.sipm_channels}
        self.sipm_data_found = False

        # Veto data
        self.pe_comp_bin_edges = self.data_aggregator.hist_calc.bin_edges_from_spec(
            config.BINS, np.array([]), config.PE_CUT)
        self.master_pe_comp_counts_2 = np.zeros(config.BINS)
        self.master_pe_comp_counts_2_or_34 = np.zeros(config.BINS)
        
        self.veto_bin_edges = np.linspace(config.VETO_RANGE[0]*0.5, config.VETO_RANGE[1], config.VETO_BINS + 1)
        self.master_veto_counts_2 = np.zeros(config.VETO_BINS)
        self.master_veto_counts_2_or_34 = np.zeros(config.VETO_BINS)
        self.veto_data_found = False
        
        # Low-light data
        self.ll_bin_edges = None
        self.master_ll_hist_counts = None
        self.ll_data_found = False

        # Highlight data
        self.hl_bin_edges = None
        self.master_hl_hist_counts = None
        self.hl_data_found = False

        # Thin veto data
        self.tv_height_bin_edges = None
        self.master_tv_muon_h_counts = None
        self.master_tv_no_co_h_counts = None
        self.tv_area_bin_edges = None
        self.master_tv_muon_a_counts = None
        self.master_tv_no_co_a_counts = None
        self.tv_data_found = False

        # Time length data
        self.total_timelength_ns = 0.0
        self.total_timelength_s = 0.0
        self.total_timelength_min = 0.0

        # Run-level veto summary data
        self.run_veto_summaries = []

    def _load_all_subjob_data(self):
        """Loops over all sub-job directories and populates the master containers."""
        for sub_dir in self.subjob_dirs:
            print(f"Processing {sub_dir.name}...")
            
            self._load_main_arrays(sub_dir)
            self._load_sipm_data(sub_dir)
            self._load_veto_data(sub_dir)
            self._load_run_level_veto_data(sub_dir)
            self._load_low_light_data(sub_dir)
            self._load_highlight_data(sub_dir)
            self._load_thin_veto_data(sub_dir)
            self._load_brn_data(sub_dir)
            self._load_time_length_data(sub_dir)

    def _load_main_arrays(self, sub_dir):
        """Load main numpy arrays (delta_t, total_pe, multiplicity)."""
        self.master_dt = self.data_aggregator.incremental_concatenate(
            self.master_dt, sub_dir / 'aggregated_delta_t.npy')
        self.master_pe = self.data_aggregator.incremental_concatenate(
            self.master_pe, sub_dir / 'aggregated_total_pe.npy')
        self.master_mult = self.data_aggregator.incremental_concatenate(
            self.master_mult, sub_dir / 'aggregated_multiplicity.npy')

    def _load_sipm_data(self, sub_dir):
        """Load and histogram SiPM data."""
        sipm_file = sub_dir / 'aggregated_sipm_area_array.pkl'
        if sipm_file.exists():
            self.sipm_data_found = True
            try:
                job_series = pd.read_pickle(sipm_file)
                job_area_data = np.array(job_series.to_list())
                if job_area_data.ndim == 1:
                    print(f"  Skipping SiPM data for {sub_dir.name}, no valid area arrays found.")
                    return
                
                for ch in self.sipm_channels:
                    if ch < job_area_data.shape[1]:
                        ch_data = job_area_data[:, ch]
                        job_counts, _ = np.histogram(ch_data, bins=self.sipm_bin_edges)
                        self.master_sipm_hist_counts[ch] += job_counts
            except Exception as e:
                print(f"  Warning: Could not process SiPM data for {sub_dir.name}. Error: {e}")

    def _load_veto_data(self, sub_dir):
        """Load and histogram veto efficiency data."""
        # Load trigger=2 data
        trig2_file = sub_dir / 'aggregated_pe_trig2.pkl'
        if trig2_file.exists():
            self.veto_data_found = True
            job_series_2 = pd.read_pickle(trig2_file)
            job_data_2 = job_series_2.to_numpy()
            
            job_counts_comp_2, _ = np.histogram(job_data_2, bins=self.pe_comp_bin_edges)
            job_counts_veto_2, _ = np.histogram(job_data_2, bins=self.veto_bin_edges)
            
            self.master_pe_comp_counts_2 += job_counts_comp_2
            self.master_veto_counts_2 += job_counts_veto_2

        # Load trigger=2 or 34 data
        trig2_34_file = sub_dir / 'aggregated_pe_trig2_or_34.pkl'
        if trig2_34_file.exists():
            self.veto_data_found = True
            job_series_2_34 = pd.read_pickle(trig2_34_file)
            job_data_2_34 = job_series_2_34.to_numpy()

            job_counts_comp_2_34, _ = np.histogram(job_data_2_34, bins=self.pe_comp_bin_edges)
            job_counts_veto_2_34, _ = np.histogram(job_data_2_34, bins=self.veto_bin_edges)

            self.master_pe_comp_counts_2_or_34 += job_counts_comp_2_34
            self.master_veto_counts_2_or_34 += job_counts_veto_2_34

    def _parse_run_start_datetime(self, run_dir_name, run_start_time_str):
        """Parse run start datetime from summary string or run directory name."""
        candidate_strings = []
        if isinstance(run_start_time_str, str) and run_start_time_str != 'no_ts':
            candidate_strings.append(run_start_time_str)

        if '_' in run_dir_name:
            suffix = run_dir_name.split('_', 1)[1]
            if suffix and suffix != 'no_ts':
                candidate_strings.append(suffix)

        for candidate in candidate_strings:
            try:
                return datetime.strptime(candidate, '%Y%m%d-%H')
            except ValueError:
                continue
        return None

    def _load_run_level_veto_data(self, sub_dir):
        """Load per-run average veto efficiency summaries from run folders."""
        for run_dir in sorted(sub_dir.glob('run*')):
            if not run_dir.is_dir():
                continue

            summary_file = run_dir / 'run_veto_summary.json'
            if not summary_file.exists():
                continue

            try:
                with open(summary_file, 'r') as f:
                    info = json.load(f)

                run_number = info.get('run')
                avg_eff = info.get('average_efficiency', np.nan)
                avg_err = info.get('average_efficiency_error', np.nan)
                run_start_str = info.get('run_start_time', 'no_ts')
                run_dt = self._parse_run_start_datetime(run_dir.name, run_start_str)

                try:
                    avg_eff_val = float(avg_eff)
                except (TypeError, ValueError):
                    avg_eff_val = np.nan
                try:
                    avg_err_val = float(avg_err)
                except (TypeError, ValueError):
                    avg_err_val = np.nan

                mu1_values = np.asarray(info.get('mu1_values', [np.nan] * 12), dtype=float)
                if mu1_values.size != 12:
                    mu1_values = np.full(12, np.nan)

                hl_peak = np.asarray(info.get('highlight_peak_pe', [np.nan] * 12), dtype=float)
                if hl_peak.size != 12:
                    hl_peak = np.full(12, np.nan)

                hl_peak_err = np.asarray(info.get('highlight_peak_pe_err', [np.nan] * 12), dtype=float)
                if hl_peak_err.size != 12:
                    hl_peak_err = np.full(12, np.nan)

                try:
                    hl_avg_val = float(info.get('highlight_avg_pe', np.nan))
                except (TypeError, ValueError):
                    hl_avg_val = np.nan

                if run_number is None:
                    continue

                self.run_veto_summaries.append({
                    'run': int(run_number),
                    'run_dir': run_dir.name,
                    'run_start_time': run_start_str,
                    'run_datetime': run_dt,
                    'average_efficiency': avg_eff_val if np.isfinite(avg_eff_val) else np.nan,
                    'average_efficiency_error': avg_err_val if np.isfinite(avg_err_val) else np.nan,
                    'valid_bin_count': int(info.get('valid_bin_count', 0)),
                    'total_trig2': int(info.get('total_trig2', 0)),
                    'total_trig2_or_34': int(info.get('total_trig2_or_34', 0)),
                    'mu1_values': mu1_values,
                    'highlight_peak_pe': hl_peak,
                    'highlight_peak_pe_err': hl_peak_err,
                    'highlight_avg_pe': hl_avg_val if np.isfinite(hl_avg_val) else np.nan,
                })
            except Exception as e:
                print(f"  Warning: Could not read run-level veto summary from {summary_file}. Error: {e}")

    def _load_low_light_data(self, sub_dir):
        """Load and sum low-light histogram data."""
        ll_file = sub_dir / 'aggregated_low_light_hists.pkl'
        if ll_file.exists():
            self.ll_data_found = True
            try:
                with open(ll_file, 'rb') as f:
                    ll_data = pickle.load(f)
                job_ll_counts = ll_data['counts']
                
                if self.master_ll_hist_counts is None:
                    self.master_ll_hist_counts = job_ll_counts.copy()
                    self.ll_bin_edges = ll_data['edges']
                else:
                    for ch in self.master_ll_hist_counts.keys():
                        self.master_ll_hist_counts[ch] += job_ll_counts.get(ch, 0)
                        
            except Exception as e:
                print(f"  Warning: Could not process Low-Light data for {sub_dir.name}. Error: {e}")

    def _load_highlight_data(self, sub_dir):
        """Load and sum highlight histogram data."""
        hl_file = sub_dir / 'aggregated_highlight_hists.pkl'
        if hl_file.exists():
            self.hl_data_found = True
            try:
                with open(hl_file, 'rb') as f:
                    hl_data = pickle.load(f)
                job_hl_counts = hl_data['counts']

                if self.master_hl_hist_counts is None:
                    self.master_hl_hist_counts = {ch: np.asarray(job_hl_counts.get(ch, 0), dtype=float).copy() for ch in range(12)}
                    self.hl_bin_edges = np.asarray(hl_data['edges'], dtype=float)
                else:
                    for ch in range(12):
                        self.master_hl_hist_counts[ch] += np.asarray(job_hl_counts.get(ch, 0), dtype=float)
            except Exception as e:
                print(f"  Warning: Could not process Highlight data for {sub_dir.name}. Error: {e}")

    def _load_thin_veto_data(self, sub_dir):
        """Load and sum thin veto histogram data."""
        tv_file = sub_dir / 'aggregated_thin_veto_hists.pkl'
        if tv_file.exists():
            self.tv_data_found = True
            try:
                with open(tv_file, 'rb') as f:
                    tv_data = pickle.load(f)
                job_tv_counts = tv_data['counts']
                job_tv_edges = tv_data['edges']

                if self.master_tv_muon_h_counts is None:
                    self.master_tv_muon_h_counts = job_tv_counts.get('muon_h', 0).copy()
                    self.master_tv_muon_a_counts = job_tv_counts.get('muon_a', 0).copy()
                    self.master_tv_no_co_h_counts = job_tv_counts.get('no_co_h', 0).copy()
                    self.master_tv_no_co_a_counts = job_tv_counts.get('no_co_a', 0).copy()
                    self.tv_height_bin_edges = job_tv_edges.get('muon_h', job_tv_edges.get('no_co_h'))
                    self.tv_area_bin_edges = job_tv_edges.get('muon_a', job_tv_edges.get('no_co_a'))
                else:
                    self.master_tv_muon_h_counts += job_tv_counts.get('muon_h', 0)
                    self.master_tv_muon_a_counts += job_tv_counts.get('muon_a', 0)
                    self.master_tv_no_co_h_counts += job_tv_counts.get('no_co_h', 0)
                    self.master_tv_no_co_a_counts += job_tv_counts.get('no_co_a', 0)

            except Exception as e:
                print(f"  Warning: Could not process Thin Veto data for {sub_dir.name}. Error: {e}")

    def _load_brn_data(self, sub_dir):
        """Load BRN analysis data."""
        brn_file = sub_dir / 'aggregated_brn_channel_data.pkl'
        if brn_file.exists():
            try:
                with open(brn_file, 'rb') as f:
                    self.all_brn_data.extend(pickle.load(f))
            except Exception as e:
                print(f"Warning: Could not load BRN data from {sub_dir.name}. Error: {e}")

    def _load_time_length_data(self, sub_dir):
        """Load and sum time length data from sub-job directory."""
        json_file = sub_dir / "subjob_time_length.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    self.total_timelength_ns += data.get("timelength_ns", 0.0)
                    self.total_timelength_s += data.get("timelength_s", 0.0)
                    self.total_timelength_min += data.get("timelength_min", 0.0)
            except Exception as e:
                print(f"  Warning: Could not read time length from {json_file}. Error: {e}")
        else:
            # Fallback to summing individual runs if subjob file doesn't exist
            print(f"  Warning: {json_file} not found. Falling back to summing individual runs.")
            for run_dir in sub_dir.glob("run*"):
                if run_dir.is_dir():
                    run_json_file = run_dir / "time_length.json"
                    if run_json_file.exists():
                        try:
                            with open(run_json_file, 'r') as f:
                                data = json.load(f)
                                self.total_timelength_ns += data.get("timelength_ns", 0.0)
                                self.total_timelength_s += data.get("timelength_s", 0.0)
                                self.total_timelength_min += data.get("timelength_min", 0.0)
                        except Exception as e:
                            print(f"  Warning: Could not read time length from {run_json_file}. Error: {e}")

    def _save_total_time_length(self):
        """Save the total aggregated time length."""
        data = {
            "total_timelength_ns": self.total_timelength_ns,
            "total_timelength_s": self.total_timelength_s,
            "total_timelength_min": self.total_timelength_min,
            "total_timelength_hours": self.total_timelength_min / 60.0,
            "total_timelength_days": self.total_timelength_min / (60.0 * 24.0)
        }
        output_file = self.master_output_dir / "total_time_length.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Total time length saved to {output_file}")
        print(f"Total time: {self.total_timelength_min:.2f} minutes ({self.total_timelength_min/60.0:.2f} hours)")
        print(f"Total time: {self.total_timelength_min/(60.0*24.0):.2f} days")

    def _generate_master_plots(self):
        """Uses the populated master containers to generate all plots."""
        self._generate_main_plots()
        self._generate_veto_plots()
        self._generate_veto_efficiency_evolution_plot()
        self._generate_mu1_evolution_plot()
        self._generate_low_light_plots()
        self._generate_highlight_plots()
        self._generate_highlight_evolution_plot()
        self._generate_sipm_plots()
        self._generate_thin_veto_plots()
        self._generate_brn_plots()

    def _prepare_evolution_x(self, run_df):
        """Prepare x-axis values for evolution plots with date-first fallback to run number."""
        has_datetime = run_df['run_datetime'].notna().any()
        if has_datetime:
            run_df = run_df.sort_values('run_datetime')
            x = run_df['run_datetime'].to_list()
            x_label = 'Run Start Time'
            use_dates = True
        else:
            run_df = run_df.sort_values('run')
            x = run_df['run'].to_numpy()
            x_label = 'Run Number (start time unavailable)'
            use_dates = False
        return run_df, x, x_label, use_dates

    def _format_evolution_xaxis(self, ax, run_df, use_dates):
        """Apply adaptive x ticks with at most 6 major ticks."""
        if use_dates:
            locator = mdates.AutoDateLocator(maxticks=6)
            formatter = mdates.DateFormatter('%Y-%m-%d\n%H:00')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.gcf().autofmt_xdate(rotation=0, ha='center')
        else:
            max_ticks = min(6, len(run_df))
            if max_ticks > 0:
                idx = np.unique(np.linspace(0, len(run_df) - 1, max_ticks, dtype=int))
                tick_x = run_df['run'].to_numpy()[idx]
                ax.set_xticks(tick_x)

    def _generate_main_plots(self):
        """Generate delta_t, total_pe, and correlation plots."""
        if self.master_dt is not None:
            print("Aggregating delta_t, total_pe, and fitting for tau...")
            
            master_aggregated_data = {
                'delta_t': [self.master_dt], 
                'total_pe': [self.master_pe], 
                'multiplicity': [self.master_mult]
            }
            aggregate_plots(
                master_aggregated_data, config.DELTA_T_CUT, config.PE_CUT, config.BINS, 
                config.TAU_FIT_WINDOW, self.master_output_dir, self.m1_or_m2, self.agg_label, 
                config.LOGSCALE_DT_AGG, config.LOGSCALE_PE_AGG, config.DO_TAU_FIT
            )
            
            master_corr_df = pd.DataFrame({
                'delta_t': self.master_dt, 
                'total_pe': self.master_pe, 
                'multiplicity': self.master_mult
            })
            self.plotter.plot_correlation_maps(master_corr_df, self.master_output_dir, 
                                             self.agg_label, self.m1_or_m2)

    def _generate_veto_plots(self):
        """Generate veto efficiency plots."""
        if self.veto_data_found:
            print("Aggregating veto efficiency data...")
            
            if self.master_pe_comp_counts_2.sum() > 0 or self.master_pe_comp_counts_2_or_34.sum() > 0:
                self.binned_plotter.plot_histogram_from_binned_data(
                    {'Trig=2 or 34': self.master_pe_comp_counts_2_or_34, 'Trig=2': self.master_pe_comp_counts_2},
                    self.pe_comp_bin_edges,
                    self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_total_pe_comparison_master.png",
                    f'Total PE Comparison {self.agg_label}', 'Total P.E.', self.m1_or_m2, logscale=True
                )
            else:
                print("No events for Master Total PE comparison; skipping plot.")

            veto_img_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_veto_efficiency_master.png"
            veto_pkl_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_veto_efficiency_master.pkl"
            self.binned_plotter.plot_veto_efficiency_from_binned_data(
                self.master_veto_counts_2, self.master_veto_counts_2_or_34,
                self.veto_bin_edges, config.VETO_RANGE,
                veto_img_path, veto_pkl_path, f"Veto Efficiency {self.agg_label}", self.m1_or_m2
            )
        else:
            print("No Veto Efficiency data found.")

    def _generate_veto_efficiency_evolution_plot(self):
        """Generate run-by-run average veto efficiency evolution plot vs real date/time."""
        if not self.run_veto_summaries:
            print("No per-run veto summary data found for evolution plot.")
            return

        run_df = pd.DataFrame(self.run_veto_summaries)
        run_df = run_df.dropna(subset=['average_efficiency'])
        if run_df.empty:
            print("Run-level veto summaries are present, but no valid average efficiency values were found.")
            return

        run_df, x, x_label, use_dates = self._prepare_evolution_x(run_df)

        y = run_df['average_efficiency'].to_numpy()
        yerr = run_df['average_efficiency_error'].to_numpy()
        finite_err = np.where(np.isfinite(yerr), yerr, 0.0)

        plt.figure(figsize=(12, 6))
        plt.errorbar(
            x, y, yerr=finite_err,
            fmt='o-', markersize=4, linewidth=1,
            capsize=2, color='navy', ecolor='steelblue',
            label='Run average veto efficiency'
        )
        plt.ylabel('Average Veto Efficiency')
        plt.xlabel(x_label)
        plt.title(f'Veto Efficiency Evolution by Run ({self.agg_label}, {self.m1_or_m2})')
        plt.grid(True, which='major', linestyle='-', linewidth=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()

        ax = plt.gca()
        self._format_evolution_xaxis(ax, run_df, use_dates)

        plt.legend()
        plt.tight_layout()

        img_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_veto_efficiency_evolution.png"
        pkl_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_veto_efficiency_evolution.pkl"
        csv_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_veto_efficiency_evolution.csv"

        plt.savefig(img_path)
        plt.close()

        out_df = run_df[['run', 'run_dir', 'run_start_time', 'average_efficiency', 'average_efficiency_error',
                         'valid_bin_count', 'total_trig2', 'total_trig2_or_34']].copy()
        if use_dates:
            out_df['run_datetime'] = pd.to_datetime(run_df['run_datetime']).dt.strftime('%Y-%m-%d %H:00')
        else:
            out_df['run_datetime'] = pd.NA

        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'run_summaries': self.run_veto_summaries,
                'plot_data': out_df.to_dict(orient='list')
            }, f)
        out_df.to_csv(csv_path, index=False)
        print(f"Veto efficiency evolution plot saved to {img_path}")
        print(f"Veto efficiency evolution data saved to {pkl_path}")

    def _generate_mu1_evolution_plot(self):
        """Generate mu1 evolution for all 12 PMT channels."""
        if not self.run_veto_summaries:
            print("No per-run summary data found for mu1 evolution plot.")
            return

        run_df = pd.DataFrame(self.run_veto_summaries)
        if 'mu1_values' not in run_df.columns:
            print("No mu1 values found in run summaries.")
            return

        run_df = run_df[run_df['mu1_values'].apply(lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 12)]
        if run_df.empty:
            print("Run summaries present, but no valid mu1 arrays were found.")
            return

        run_df, x, x_label, use_dates = self._prepare_evolution_x(run_df)

        plt.figure(figsize=(13, 7))
        cmap = plt.cm.get_cmap('tab20', 12)
        for ch in range(12):
            y = run_df['mu1_values'].apply(lambda arr: float(np.asarray(arr, dtype=float)[ch])).to_numpy()
            plt.plot(x, y, marker='o', linewidth=1.2, markersize=3.5, color=cmap(ch), label=f'PMT {ch}')

        plt.ylabel('$\\mu_1$ Area (ADC)')
        plt.xlabel(x_label)
        plt.title(f'Low-Light $\\mu_1$ Evolution by Run ({self.agg_label}, {self.m1_or_m2})')
        plt.grid(True, which='major', linestyle='-', linewidth=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        ax = plt.gca()
        self._format_evolution_xaxis(ax, run_df, use_dates)
        plt.legend(ncol=3, fontsize='small')
        plt.tight_layout()

        img_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_mu1_evolution.png"
        pkl_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_mu1_evolution.pkl"
        csv_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_mu1_evolution.csv"
        plt.savefig(img_path)
        plt.close()

        out_df = run_df[['run', 'run_dir', 'run_start_time']].copy()
        mu1_matrix = np.vstack(run_df['mu1_values'].apply(lambda arr: np.asarray(arr, dtype=float)).to_numpy())
        for ch in range(12):
            out_df[f'mu1_ch{ch}'] = mu1_matrix[:, ch]
        if use_dates:
            out_df['run_datetime'] = pd.to_datetime(run_df['run_datetime']).dt.strftime('%Y-%m-%d %H:00')
        else:
            out_df['run_datetime'] = pd.NA

        with open(pkl_path, 'wb') as f:
            pickle.dump({'plot_data': out_df.to_dict(orient='list')}, f)
        out_df.to_csv(csv_path, index=False)
        print(f"mu1 evolution plot saved to {img_path}")

    def _generate_highlight_plots(self):
        """Generate aggregated highlight PMT PE fit plots."""
        if self.hl_data_found and self.master_hl_hist_counts is not None and self.hl_bin_edges is not None:
            print("Plotting aggregated highlight PMT P.E. data...")
            hl_cfg = getattr(config, 'HIGHLIGHT_FIT_CONFIG', {}) or {}
            hist_range = tuple(hl_cfg.get('hist_range', (float(self.hl_bin_edges[0]), float(self.hl_bin_edges[-1]))))
            self.binned_plotter.fit_and_plot_highlight_from_binned_data(
                self.master_hl_hist_counts,
                self.hl_bin_edges,
                self.master_output_dir,
                self.agg_label,
                self.m1_or_m2,
                hist_range=hist_range,
            )
        else:
            print("No Highlight data found to plot.")

    def _generate_highlight_evolution_plot(self):
        """Generate highlight peak evolution for 12 PMTs + average line."""
        if not self.run_veto_summaries:
            print("No per-run summary data found for highlight evolution plot.")
            return

        run_df = pd.DataFrame(self.run_veto_summaries)
        if 'highlight_peak_pe' not in run_df.columns:
            print("No highlight peak values found in run summaries.")
            return

        run_df = run_df[run_df['highlight_peak_pe'].apply(lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 12)]
        if run_df.empty:
            print("Run summaries present, but no valid highlight peak arrays were found.")
            return

        run_df, x, x_label, use_dates = self._prepare_evolution_x(run_df)
        peak_matrix = np.vstack(run_df['highlight_peak_pe'].apply(lambda arr: np.asarray(arr, dtype=float)).to_numpy())
        err_matrix = np.vstack(run_df['highlight_peak_pe_err'].apply(lambda arr: np.asarray(arr, dtype=float)).to_numpy()) if 'highlight_peak_pe_err' in run_df.columns else np.full_like(peak_matrix, np.nan)
        avg_vals = run_df['highlight_avg_pe'].to_numpy(dtype=float) if 'highlight_avg_pe' in run_df.columns else np.nanmean(peak_matrix, axis=1)

        plt.figure(figsize=(13, 7))
        cmap = plt.cm.get_cmap('tab20', 12)
        for ch in range(12):
            y = peak_matrix[:, ch]
            yerr = np.where(np.isfinite(err_matrix[:, ch]), err_matrix[:, ch], 0.0)
            plt.errorbar(x, y, yerr=yerr, fmt='o-', markersize=3.0, linewidth=1.0, capsize=1.5,
                         color=cmap(ch), alpha=0.9, label=f'PMT {ch}')

        plt.plot(x, avg_vals, 'k--', linewidth=2.0, marker='s', markersize=4.0, label='Average (12 PMTs)')
        plt.ylabel('Highlight Peak (P.E.)')
        plt.xlabel(x_label)
        plt.title(f'Highlight Peak Evolution by Run ({self.agg_label}, {self.m1_or_m2})')
        plt.grid(True, which='major', linestyle='-', linewidth=0.7)
        plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
        plt.minorticks_on()
        ax = plt.gca()
        self._format_evolution_xaxis(ax, run_df, use_dates)
        plt.legend(ncol=3, fontsize='small')
        plt.tight_layout()

        img_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_highlight_peak_evolution.png"
        pkl_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_highlight_peak_evolution.pkl"
        csv_path = self.master_output_dir / f"{self.filename_label}_{self.m1_or_m2}_highlight_peak_evolution.csv"
        plt.savefig(img_path)
        plt.close()

        out_df = run_df[['run', 'run_dir', 'run_start_time']].copy()
        for ch in range(12):
            out_df[f'peak_ch{ch}'] = peak_matrix[:, ch]
            out_df[f'peak_err_ch{ch}'] = err_matrix[:, ch]
        out_df['peak_avg_12ch'] = avg_vals
        if use_dates:
            out_df['run_datetime'] = pd.to_datetime(run_df['run_datetime']).dt.strftime('%Y-%m-%d %H:00')
        else:
            out_df['run_datetime'] = pd.NA

        with open(pkl_path, 'wb') as f:
            pickle.dump({'plot_data': out_df.to_dict(orient='list')}, f)
        out_df.to_csv(csv_path, index=False)
        print(f"Highlight peak evolution plot saved to {img_path}")

    def _generate_low_light_plots(self):
        """Generate low-light fit plots."""
        if self.ll_data_found:
            print("Plotting aggregated Low-Light data...")
            self.binned_plotter.fit_and_plot_low_light_from_binned_data(
                self.master_ll_hist_counts,
                self.ll_bin_edges,
                self.master_output_dir,
                self.agg_label,
                self.m1_or_m2,
                hist_range=config.LOW_LIGHT_FIT_RANGE
            )
        else:
            print("No Low-Light data found to plot.")

    def _generate_sipm_plots(self):
        """Generate SiPM histogram plots."""
        if self.sipm_data_found:
            print("Plotting aggregated SiPM area data...")
            self.binned_plotter.plot_sipm_histograms_from_binned_data(
                self.master_sipm_hist_counts, 
                self.sipm_bin_edges, 
                self.master_output_dir, 
                self.agg_label, 
                self.m1_or_m2, 
                config.SIPM_HIST_CONFIG
            )
        else:
            print("No SiPM data found to plot.")

    def _generate_thin_veto_plots(self):
        """Generate thin veto comparison plots."""
        if self.master_tv_muon_h_counts is not None:
            print("Aggregating Thin Veto data...")

            height_img_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_thin_veto_height_comparison_master.png'
            self.binned_plotter.plot_normalized_histogram_comparison_from_binned_data(
                self.master_tv_muon_h_counts, 'Muon Events (Coincidence)', 
                self.master_tv_no_co_h_counts, 'All Triggered Events',
                self.tv_height_bin_edges, height_img_path, 
                f'Master Thin Veto Height Comparison - {self.agg_label}',
                'Pulse Height (ADC)', self.m1_or_m2
            )
            
            area_img_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_thin_veto_area_comparison_master.png'
            self.binned_plotter.plot_normalized_histogram_comparison_from_binned_data(
                self.master_tv_muon_a_counts, 'Muon Events (Coincidence)', 
                self.master_tv_no_co_a_counts, 'All Triggered Events',
                self.tv_area_bin_edges, area_img_path, 
                f'Master Thin Veto Area Comparison - {self.agg_label}',
                'Pulse Area (ADC)', self.m1_or_m2
            )

    def _generate_brn_plots(self):
        """Generate BRN analysis plots."""
        if self.all_brn_data:
            print("Aggregating BRN Analysis data...")
            master_brn_data = self.data_aggregator.merge_channel_data_dicts(self.all_brn_data)
            if master_brn_data:
                self._plot_brn_histograms(master_brn_data)
            else:
                print("BRN data was found but failed to merge. Skipping master plots.")
    
    def _plot_brn_histograms(self, channel_data):
        """
        Plot BRN (Beam-Related Neutron) analysis histograms for SiPM channels.
        Creates multi-panel plots showing delta_t and area distributions for channels 12-21.
        
        Args:
            channel_data: Dictionary with channel numbers as keys and sub-dicts containing
                         'delta_t' and 'area' numpy arrays as values.
        """
        if not channel_data:
            print("No BRN channel data to plot.")
            return
        
        print(f"Plotting BRN histograms for {len(channel_data)} channels...")
        
        # Extract BRN configuration
        brn_channels = config.BRN_SIPM_CHANNELS
        delta_t_range = config.BRN_DELTA_T_RANGE
        area_range = config.BRN_HIST_CONFIG['area_range']
        area_bins = config.BRN_HIST_CONFIG['area_bins']
        
        # Calculate delta_t bins based on range and bin width
        delta_t_bin_width = config.BRN_DELTA_T_BIN_WIDTH_NS
        n_delta_t_bins = int((delta_t_range[1] - delta_t_range[0]) / delta_t_bin_width)
        delta_t_bins = np.linspace(delta_t_range[0], delta_t_range[1], n_delta_t_bins + 1)
        
        area_bin_edges = np.linspace(area_range[0], area_range[1], area_bins + 1)
        
        # --- Plot Delta_t Histograms ---
        fig_dt, axes_dt = plt.subplots(3, 4, figsize=(20, 15))
        fig_dt.suptitle(f'BRN Delta_t Distribution - {self.agg_label} ({self.m1_or_m2})', fontsize=16)
        axes_dt = axes_dt.flatten()
        
        brn_delta_t_data = {}
        
        for i, ch in enumerate(brn_channels):
            ax = axes_dt[i]
            if ch in channel_data and 'delta_t' in channel_data[ch]:
                dt_data = channel_data[ch]['delta_t']
                if dt_data.size > 0:
                    counts, _ = np.histogram(dt_data, bins=delta_t_bins)
                    total_events = len(dt_data)
                    
                    ax.step(delta_t_bins, np.append(counts, counts[-1]), where='post', 
                           color='navy', label=f"N = {total_events:.0f}")
                    
                    brn_delta_t_data[ch] = {'counts': counts, 'edges': delta_t_bins}
                    ax.set_title(f'SiPM Channel {ch}')
                    ax.set_xlabel('Delta_t (ns)')
                    ax.set_ylabel('Events')
                    ax.grid(True, which='both', linestyle=':')
                    # ax.set_yscale('log')
                    ax.set_xlim(delta_t_range)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'Channel {ch}\nNo Events', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, f'Channel {ch}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
        
        # Hide unused subplots
        for i in range(len(brn_channels), len(axes_dt)):
            axes_dt[i].set_axis_off()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        dt_img_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_brn_delta_t_master.png'
        dt_pkl_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_brn_delta_t_master.pkl'
        plt.savefig(dt_img_path)
        self.file_handler.save_pickle(brn_delta_t_data, dt_pkl_path)
        print(f"BRN Delta_t histograms saved to {dt_img_path}")
        plt.close(fig_dt)
        
        # --- Plot Area Histograms ---
        fig_area, axes_area = plt.subplots(3, 4, figsize=(20, 15))
        fig_area.suptitle(f'BRN Area Distribution - {self.agg_label} ({self.m1_or_m2})', fontsize=16)
        axes_area = axes_area.flatten()
        
        brn_area_data = {}
        
        for i, ch in enumerate(brn_channels):
            ax = axes_area[i]
            if ch in channel_data and 'area' in channel_data[ch]:
                area_data = channel_data[ch]['area']
                if area_data.size > 0:
                    counts, _ = np.histogram(area_data, bins=area_bin_edges)
                    total_events = len(area_data)
                    
                    ax.step(area_bin_edges, np.append(counts, counts[-1]), where='post', 
                           color='darkcyan', label=f"N = {total_events:.0f}")
                    
                    brn_area_data[ch] = {'counts': counts, 'edges': area_bin_edges}
                    ax.set_title(f'SiPM Channel {ch}')
                    ax.set_xlabel('Area (ADC)')
                    ax.set_ylabel('Events')
                    ax.grid(True, which='both', linestyle=':')
                    # ax.set_yscale('log')
                    ax.set_xlim(area_range)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f'Channel {ch}\nNo Events', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_axis_off()
            else:
                ax.text(0.5, 0.5, f'Channel {ch}\nNo Data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
        
        # Hide unused subplots
        for i in range(len(brn_channels), len(axes_area)):
            axes_area[i].set_axis_off()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        area_img_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_brn_area_master.png'
        area_pkl_path = self.master_output_dir / f'{self.filename_label}_{self.m1_or_m2}_brn_area_master.pkl'
        plt.savefig(area_img_path)
        self.file_handler.save_pickle(brn_area_data, area_pkl_path)
        print(f"BRN Area histograms saved to {area_img_path}")
        plt.close(fig_area)

    def _save_run_info(self):
        """Save configuration and run information to a text file."""
        info_file = self.master_output_dir / "run_info.txt"
        
        with open(info_file, "w") as f:
            f.write(f"Analysis Run Information\n")
            f.write(f"========================\n\n")
            f.write(f"Run Range: {self.run_range_str}\n")
            f.write(f"M1/M2: {self.m1_or_m2}\n")
            f.write(f"Number of Sub-jobs: {len(self.subjob_dirs)}\n")
            f.write(f"Top Directory: {self.top_dir}\n\n")
            
            f.write(f"Configuration Parameters\n")
            f.write(f"------------------------\n")
            f.write(f"DATA_DIR_M1: {config.DATA_DIR_M1}\n")
            f.write(f"DATA_DIR_M2: {config.DATA_DIR_M2}\n")
            f.write(f"DELTA_T_CUT: {config.DELTA_T_CUT}\n")
            f.write(f"PE_CUT: {config.PE_CUT}\n")
            f.write(f"TIME_STD_CUT: {config.TIME_STD_CUT}\n")
            f.write(f"MULTIPLICITY_SPE: {config.MULTIPLICITY_SPE}\n")
            f.write(f"MULTIPLICITY_CUT: {config.MULTIPLICITY_CUT}\n")
            f.write(f"DELTA_T_BIN_WIDTH_NS: {config.DELTA_T_BIN_WIDTH_NS}\n")
            f.write(f"BINS: {config.BINS}\n")
            f.write(f"VETO_BINS: {config.VETO_BINS}\n")
            f.write(f"VETO_RANGE: {config.VETO_RANGE}\n")
            f.write(f"TAU_FIT_WINDOW: {config.TAU_FIT_WINDOW}\n")
            f.write(f"LOW_LIGHT_FIT_RANGE: {config.LOW_LIGHT_FIT_RANGE}\n")
            f.write(f"HIGHLIGHT_FIT_CONFIG: {getattr(config, 'HIGHLIGHT_FIT_CONFIG', {})}\n")
            f.write(f"SIPM_HIST_CONFIG: {config.SIPM_HIST_CONFIG}\n")
            f.write(f"PERFORM_THIN_VETO_ANALYSIS: {config.PERFORM_THIN_VETO_ANALYSIS}\n")
            f.write(f"PERFORM_BRN_ANALYSIS: {config.PERFORM_BRN_ANALYSIS}\n")
            
            f.write(f"\nSub-job Directories Processed:\n")
            for d in self.subjob_dirs:
                f.write(f"  {d.name}\n")
            
        print(f"Run info saved to {info_file}")

    def run(self):
        """Run the full aggregation process."""
        self._load_all_subjob_data()
        self._generate_master_plots()
        self._save_total_time_length()
        self._save_run_info()
        print("\n--- Master Aggregation Complete ---")

def main():
    """Script entry point."""
    if len(sys.argv) != 2:
        print("Usage: python aggregate_master_veto.py <top_level_analysis_directory>")
        sys.exit(1)
    
    try:
        aggregator = MasterAggregator(sys.argv[1])
        aggregator.run()
    except (FileNotFoundError, Exception) as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()