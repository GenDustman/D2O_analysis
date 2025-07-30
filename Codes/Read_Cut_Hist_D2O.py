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
"""
import sys
from pathlib import Path
import pickle
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


def ensure_dir(path: Path):
    """
    Ensure that a directory exists; create it and any parent directories if necessary.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(data: dict, path: Path):
    """
    Serialize and save a Python dictionary to a pickle file.
    """
    with path.open('wb') as f:
        pickle.dump(data, f)


def plot_histogram(arrays, labels, bins, img_path, title, xlabel,
                   M1_or_M2, logscale=True, figsize=(10, 6)):
    """
    Plot one or more datasets as overlapping histograms, with consistent styling.
    """
    plt.figure(figsize=figsize)
    outputs = []
    for data, label in zip(arrays, labels):
        counts, edges, _ = plt.hist(
            data,
            bins=bins,
            alpha=0.7,
            edgecolor='black',
            label=label
        )
        outputs.append((counts, edges))
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
    plt.close()
    return outputs


def plot_sipm_histograms(df, output_dir, label, M1_or_M2, hist_bins=100, hist_range=(-50, 1500)):
    """
    Selects events with trigger bit >= 32 and plots area histograms for SiPM channels 12-21.
    Saves the histogram data to a pickle file.
    """
    sipm_events = df[df['triggerBits'] >= 32].copy()
    if sipm_events.empty:
        print(f"No SiPM events (triggerBits >= 32) found for {label}. Skipping SiPM histograms.")
        return

    area_data = np.array(sipm_events['area_array'].to_list())
    
    sipm_channels = range(12, 22)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'SiPM Channel Area (triggerBits>=32) - {label} ({M1_or_M2})', fontsize=16)
    axes = axes.flatten()
    sipm_hist_data = {}

    for i, ch in enumerate(sipm_channels):
        ax = axes[i]
        if ch < area_data.shape[1]:
            ch_data = area_data[:, ch]
            counts, edges, _ = ax.hist(ch_data, bins=hist_bins, range=hist_range, histtype='step', linewidth=1.5, color='darkcyan')
            sipm_hist_data[ch] = {'counts': counts, 'edges': edges}
            ax.set_title(f'SiPM Channel {ch}')
            ax.set_xlabel('Area (ADC)')
            ax.set_ylabel('Events')
            ax.grid(True, which='both', linestyle=':')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f'Channel {ch}\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
    
    for i in range(len(sipm_channels), len(axes)):
        axes[i].set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(output_dir)
    
    ## FILENAME CHANGE: Add M1_or_M2 to filenames
    filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
    base_filename = f'{filename_label}_{M1_or_M2}_sipm_area_histograms'
    img_save_path = output_dir / f'{base_filename}.png'
    pkl_save_path = output_dir / f'{base_filename}.pkl'
    
    plt.savefig(img_save_path)
    save_pickle(sipm_hist_data, pkl_save_path)
    print(f"SiPM histograms saved to {img_save_path}")
    print(f"SiPM histogram data saved to {pkl_save_path}")
    plt.close()


def plot_correlation_maps(df, output_dir, label, M1_or_M2):
    """
    Plots a 3x3 grid of correlation maps for delta_t, total_pe, and multiplicity.
    """
    ensure_dir(output_dir)
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
                    if h[0].max() > 0: fig.colorbar(h[3], ax=ax)

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
    
    ## FILENAME CHANGE: Add M1_or_M2 to filename
    filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
    save_path = output_dir / f'{filename_label}_{M1_or_M2}_correlation_map.png'
    
    plt.savefig(save_path)
    print(f"Correlation map saved to {save_path}")
    plt.close()


def compute_delta_t(df, muon_bits, veto_bits, mult_thresh):
    """
    Compute time differences Δt between veto events and the preceding muon event.
    """
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


def save_cut_histograms(events, delta_t_range, pe_range, bins,
                        save_dir, run_label, time_std_cut, M1_or_M2, logscale=True):
    """
    Apply sequential cuts and save errorbar histograms.
    """
    dt_min, dt_max = delta_t_range
    pe_min, pe_max = pe_range

    ensure_dir(save_dir)
    sel = events.dropna(subset=['delta_t', 'total_pe']).copy()
    print(f"{run_label}: after NaN drop: {len(sel)} events")
    sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
    print(f"{run_label}: after Δt cut: {len(sel)} events")
    sel = sel[(sel['total_pe'] >= pe_min) & (sel['total_pe'] <= pe_max)]
    print(f"{run_label}: after total_pe cut: {len(sel)} events")

    sel = sel.dropna(subset=['time_std'])
    sel = sel[sel['time_std'] < time_std_cut]
    print(f"{run_label}: after time-std < {time_std_cut} ns cut: {len(sel)} events")

    plot_correlation_maps(sel, save_dir, run_label, M1_or_M2)

    if sel.empty:
        return None, None, None

    # --- Delta T Histogram ---
    dt_bins = np.linspace(dt_min, dt_max, bins + 1)
    dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
    dt_err = np.sqrt(dt_counts)
    
    ## FILENAME CHANGE: Add M1_or_M2 to filenames
    dt_base_filename = f'delta_t_hist_{M1_or_M2}'
    save_pickle({'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err}, save_dir / f'{dt_base_filename}.pkl')
    
    plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
    plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'Δt Histogram ({M1_or_M2})')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / f'{dt_base_filename}.png'); plt.close()

    # --- Total PE Histogram ---
    pe_bins = np.linspace(pe_min, pe_max, bins + 1)
    pe_counts, pe_edges = np.histogram(sel['total_pe'], bins=pe_bins)
    pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
    peak_location = pe_centers[np.argmax(pe_counts)]
    peak = np.round(peak_location, 1)
    mean_pe = sel['total_pe'].mean()
    mean_pe_val = np.round(mean_pe, 1)
    pe_err = np.sqrt(pe_counts)

    ## FILENAME CHANGE: Add M1_or_M2 to filenames
    pe_base_filename = f'total_pe_hist_{M1_or_M2}'
    save_pickle({'hist': pe_counts, 'centers': pe_centers, 'errors': pe_err}, save_dir / f'{pe_base_filename}.pkl')

    plot_label = f'{run_label}\nMean = {mean_pe_val} p.e.'
    plt.errorbar(pe_centers, pe_counts, yerr=pe_err, fmt='o', label=plot_label)
    
    plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title(f'Total Photoelectron Histogram ({M1_or_M2})')
    plt.axvline(peak, color='red', linestyle='--', label=f'Peak = {peak} p.e.')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / f'{pe_base_filename}.png'); plt.close()

    return sel['delta_t'].values, sel['total_pe'].values, sel['multiplicity'].values


def fit_and_plot_low_light(area_data, output_dir, file_label, M1_or_M2, hist_range, hist_bins=200):
    """
    Plots and fits sum_area for channels 0-11 for low-light events (triggerbit=16).
    """
    if area_data.size == 0:
        print(f"No low-light data to process for {file_label}.")
        return np.full(12, np.nan)

    def constrained_gaussians(x, a0, mu0, sig0, a1, mu1, sig1, a2, a3):
        sig2_sq = 2 * sig1**2 - sig0**2
        sig3_sq = 3 * sig1**2 - 2 * sig0**2
        if sig2_sq < 0 or sig3_sq < 0: return np.inf
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
    ensure_dir(output_dir)
    
    ## FILENAME CHANGE: Add M1_or_M2 to filenames
    filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
    base_filename = f'{filename_label}_{M1_or_M2}_low_light_fits'
    img_save_path = output_dir / f'{base_filename}.png'
    pkl_save_path = output_dir / f'{base_filename}.pkl'
    
    plt.savefig(img_save_path)
    save_pickle(fit_results_data, pkl_save_path)
    print(f"Low-light fits saved to {img_save_path}")
    print(f"Low-light fit data saved to {pkl_save_path}")
    plt.close()
    
    return mu1_values


def calculate_total_pe(df, mu1_values):
    """
    Calculates the total photoelectrons for each event using per-channel gain.
    """
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


def process_run(run, data_dir, output_dir, delta_t_cut, pe_cut, bins,
                multiplicity_spe, multiplicity_cut, time_std_cut, logscale,
                low_light_fit_range, sipm_hist_config, M1_or_M2):
    """
    Process a single run: read data, perform calculations, and apply cuts.
    """
    print(f"--- Processing run {run} ---")
    if M1_or_M2 == 'M1':
        infile = data_dir / f"run{run}_processed_v5.root"
    elif M1_or_M2 == 'M2':
        infile = data_dir / f"run{run}_processed_H2O_v5.root"
    if not infile.exists():
        print(f"Missing file: {infile}")
        return None

    run_start_time_str = "no_ts"
    try:
        with uproot.open(infile) as f_ts:
            if 'starttime' in f_ts:
                unix_time = f_ts['starttime'].member("fVal")
                run_start_time_str = datetime.fromtimestamp(unix_time).strftime('%Y%m%d-%H')
    except Exception as e:
        print(f"Warning: Could not read start time for run {run}, using default folder name. Error: {e}")

    dfs = []
    branches = ['eventID', 'nsTime', 'triggerBits', 'area', 'peakPosition']
    for chunk in uproot.open(infile)['tree'].iterate(branches, library='ak', step_size='500 MB'):
        df = pd.DataFrame({
            'eventID': ak.to_numpy(chunk['eventID']),
            'nsTime': ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            'area_array': ak.to_list(chunk['area']),
            'peakPosition': ak.to_list(chunk['peakPosition']),
        })
        dfs.append(df)

    if not dfs: return None
    df_all = pd.concat(dfs, ignore_index=True)

    run_dir = output_dir / f"run{run}_{run_start_time_str}"
    hist_dir = run_dir / "histograms"
    cut_dir = run_dir / "cuthist"
    ll_dir = run_dir / "lowlight"
    ensure_dir(hist_dir); ensure_dir(cut_dir); ensure_dir(ll_dir)

    ## FILENAME CHANGE: Add M1_or_M2 to filename
    plot_histogram([df_all['triggerBits'].to_numpy()], ['triggerBits'],
                   np.arange(0, 36), hist_dir / f"{run}_{M1_or_M2}_triggerBits.png",
                   'Trigger Bits Distribution', 'triggerBits', M1_or_M2, logscale)

    ll_events = df_all[df_all['triggerBits'] == 16]
    low_light_area_data = np.array(ll_events['area_array'].to_list())[:, :12] if not ll_events.empty else np.array([])

    if low_light_area_data.size > 0:
        mu1_values_run = fit_and_plot_low_light(low_light_area_data, ll_dir, f'Run{run}', M1_or_M2, hist_range=low_light_fit_range)
    else:
        print(f"No low-light events for run {run}. P.E. and multiplicity calculations will fail.")
        mu1_values_run = np.full(12, np.nan)

    area_data_np = np.array(df_all['area_array'].to_list())[:, :12]
    times_data_np = np.array(df_all['peakPosition'].to_list())[:, :12]
    mu1_safe = np.where(np.isnan(mu1_values_run) | (mu1_values_run <= 0), np.inf, mu1_values_run)
    pe_per_channel = area_data_np / mu1_safe
    postmcut_mask = pe_per_channel > multiplicity_spe
    df_all['multiplicity'] = np.sum(postmcut_mask, axis=1)
    masked_times = np.where(postmcut_mask, times_data_np, np.nan)
    df_all['time_std'] = np.nanstd(masked_times, axis=1)

    df_all['total_pe'] = calculate_total_pe(df_all, mu1_values_run)
    ## FILENAME CHANGE: Add M1_or_M2 to filename
    df_all.to_pickle(run_dir / f"run{run}_{M1_or_M2}_data_with_pe.pkl")
    
    ## FILENAME CHANGE: Add M1_or_M2 to filename
    plot_histogram([df_all.loc[(df_all['triggerBits'] == 2) | (df_all['triggerBits'] == 34), 'total_pe'].dropna(), df_all.loc[df_all['triggerBits'] == 2, 'total_pe'].dropna()],
                   ['Trig=2 or 34', 'Trig=2'], np.linspace(0, 2000, bins + 1),
                   hist_dir / f"{run}_{M1_or_M2}_total_pe.png", 'Total Photoelectron Comparison', 'Total P.E.', M1_or_M2, logscale)
    
    events = compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)
    cut_results = save_cut_histograms(
        events, delta_t_cut, pe_cut, bins, cut_dir,
        f"Run {run}", time_std_cut, M1_or_M2, logscale
    )
    
    plot_sipm_histograms(df_all, run_dir, f"Run {run}", M1_or_M2, **sipm_hist_config)
    
    sipm_events_df = df_all[df_all['triggerBits'] >= 32]
    
    if cut_results:
        dt_vals, pe_vals, mult_vals = cut_results
        return dt_vals, pe_vals, mult_vals, low_light_area_data, sipm_events_df
    else:
        return None, None, None, low_light_area_data, sipm_events_df


def aggregate_plots(aggregated, delta_t_cut, pe_cut, bins,
                    fit_window, output_dir, M1_or_M2, label, logscale_dt, logscale_pe,
                    perform_fit=True):
    """
    Generate aggregated histograms with simple Poisson errors.
    """
    ensure_dir(output_dir)
    dt_min, dt_max = delta_t_cut
    all_dt = np.concatenate(aggregated['delta_t']) if aggregated['delta_t'] else np.array([])
    
    ## FILENAME CHANGE: Create a filename-safe label from the run range
    filename_label = label.replace(" ", "_").replace("-", "_")

    if all_dt.size:
        dt_bins = np.linspace(dt_min, dt_max, bins + 1)
        hist_dt, dt_edges = np.histogram(all_dt, bins=dt_bins)
        dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
        dt_err = np.sqrt(hist_dt)

        pickle_data = {'centers': dt_centers, 'hist': hist_dt, 'errors': dt_err}
        plt.errorbar(dt_centers, hist_dt, yerr=dt_err, fmt='o', label='Data')

        if perform_fit:
            t_low, t_high = fit_window
            mask = (dt_centers >= t_low) & (dt_centers <= t_high) & (hist_dt > 0)
            if np.any(mask):
                fit_x = dt_centers[mask]
                fit_y = np.log(hist_dt[mask])
                try:
                    (slope, intercept), cov = np.polyfit(fit_x, fit_y, 1, w=np.sqrt(hist_dt[mask]), cov=True)
                    slope_err = np.sqrt(cov[0, 0])
                    tau = -1.0 / slope
                    tau_err = slope_err / (slope**2)
                    fit_line = np.exp(intercept + slope * dt_centers)
                    plt.plot(dt_centers, fit_line, '--', label=f'Fit τ={tau:.1f}±{tau_err:.1f} ns')
                    plt.axvspan(t_low, t_high, color='gray', alpha=0.2, label='Fit Range')
                    pickle_data['tau'] = tau
                    pickle_data['tau_err'] = tau_err
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"Warning: Could not perform the fit. Error: {e}")
            else:
                print("Warning: No data in the specified fit window. Skipping the fit.")

        plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'{label} Δt ({M1_or_M2})')
        if logscale_dt: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        
        ## FILENAME CHANGE: Use new label and M1_or_M2 for filenames
        dt_base_filename = f'{filename_label}_{M1_or_M2}_delta_t'
        plt.savefig(output_dir / f'{dt_base_filename}.png'); plt.close()
        save_pickle(pickle_data, output_dir / f'{dt_base_filename}.pkl')

    all_pe = np.concatenate(aggregated['total_pe']) if aggregated['total_pe'] else np.array([])
    if all_pe.size:
        pe_min, pe_max = pe_cut
        pe_bins = np.linspace(pe_min, pe_max, bins + 1)
        hist_pe, pe_edges = np.histogram(all_pe, bins=pe_bins)
        pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
        peak_location = pe_centers[np.argmax(hist_pe)]
        peak = np.round(peak_location, 1)
        mean_pe = all_pe.mean()
        mean_pe_val = np.round(mean_pe, 1)
        pe_err = np.sqrt(hist_pe)

        plot_label = f'{label}\nMean = {mean_pe_val} p.e.'
        plt.errorbar(pe_centers, hist_pe, yerr=pe_err, fmt='o', label=plot_label)
        
        plt.axvline(peak, color='red', linestyle='--', label=f'Peak = {peak} p.e.')
        plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title(f'{label} Total Photoelectrons ({M1_or_M2})')
        if logscale_pe: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        
        ## FILENAME CHANGE: Use new label and M1_or_M2 for filenames
        pe_base_filename = f'{filename_label}_{M1_or_M2}_total_pe'
        plt.savefig(output_dir / f'{pe_base_filename}.png'); plt.close()
        save_pickle({'centers': pe_centers, 'hist': hist_pe, 'errors': pe_err},
                    output_dir / f'{pe_base_filename}.pkl')


def main():
    """
    Entry point: parse arguments, print configuration,
    loop over runs, and generate aggregated histograms.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_run> <end_run>")
        sys.exit(1)
    start_run, end_run = map(int, sys.argv[1:])

    # --- Configuration Parameters ---
    M1_or_M2 = 'M2'
    delta_t_cut = (0, 10000)
    pe_cut = (0, 1000)
    bins = 100
    multiplicity_spe = 1.0
    multiplicity_cut = 10
    time_std_cut = 2.5 * 16
    logscale = True
    logscale_dt = True
    logscale_pe = False
    do_tau_fit = True
    tau_fit_window = (2500, 10000)
    low_light_fit_range = (-50, 400)
    
    sipm_hist_config = {
        'hist_bins': 100,
        'hist_range': (-50, 4000)
    }
    # --------------------------------
    
    dt_min, dt_max = delta_t_cut
    pe_min, pe_max = pe_cut

    if M1_or_M2 == 'M1':
        data_dir = Path('/raid1/genli/Data_D2O/M1_data')
    elif M1_or_M2 == 'M2':
        data_dir = Path('/raid1/genli/Data_D2O/M2_data')
    else:
        raise ValueError("M1_or_M2 must be 'M1' or 'M2'.")

    first_run_file = data_dir / (f"run{start_run}_processed_v5.root" if M1_or_M2 == 'M1' else f"run{start_run}_processed_H2O_v5.root")
    run_start_time_str = ""
    try:
        with uproot.open(first_run_file) as f:
            if 'starttime' in f:
                unix_time = f['starttime'].member("fVal")
                run_start_time_str = datetime.fromtimestamp(unix_time).strftime('%Y%m%d-%H')
            else:
                print(f"Warning: 'starttime' not found in {first_run_file}. Using current time for folder name.")
                run_start_time_str = datetime.now().strftime('%Y%m%d-%H')
    except FileNotFoundError:
        print(f"Error: First run file not found at {first_run_file}. Cannot determine start time to create output folder.")
        sys.exit(1)
    
    output_dir = data_dir / (
        f"runs_{start_run}_{end_run}_{M1_or_M2}_{run_start_time_str}"
        f"_dt{dt_min}-{dt_max}_pe{pe_min}-{pe_max}"
        f"_mspe{multiplicity_spe}_mcut{multiplicity_cut}"
    )
    ensure_dir(output_dir)

    print("=== Configuration ===")
    print(f"Analysis type: {M1_or_M2}")
    print(f"Runs: {start_run} to {end_run}")
    print(f"Output Directory: {output_dir}")
    print(f"Δt cut: {delta_t_cut} ns")
    print(f"Photoelectron cut: {pe_cut} P.E.")
    print(f"Multiplicity SPE threshold: {multiplicity_spe}")
    print(f"Multiplicity cut: {multiplicity_cut}")
    print(f"Time-std cut: < {time_std_cut} ns")
    print(f"SiPM Area Histogram Bins: {sipm_hist_config['hist_bins']}, Range: {sipm_hist_config['hist_range']} ADC")
    print("======================")

    aggregated = {
        'delta_t': [], 'total_pe': [], 'multiplicity': [],
        'low_light_areas': [], 'sipm_events': []
    }

    for run in range(start_run, end_run + 1):
        result = process_run(
            run, data_dir, output_dir, delta_t_cut, pe_cut, bins,
            multiplicity_spe, multiplicity_cut, time_std_cut,
            logscale, low_light_fit_range, sipm_hist_config, M1_or_M2
        )
        if result:
            dt_vals, pe_vals, mult_vals, ll_areas, sipm_df = result
            if dt_vals is not None: aggregated['delta_t'].append(dt_vals)
            if pe_vals is not None: aggregated['total_pe'].append(pe_vals)
            if mult_vals is not None: aggregated['multiplicity'].append(mult_vals)
            if ll_areas.size > 0: aggregated['low_light_areas'].append(ll_areas)
            if not sipm_df.empty: aggregated['sipm_events'].append(sipm_df)
    
    agg_label = f"Runs {start_run}-{end_run}"

    # --- Aggregated PMT Analysis ---
    aggregate_plots(
        aggregated, delta_t_cut, pe_cut, bins, tau_fit_window,
        output_dir, M1_or_M2, agg_label, logscale_dt, logscale_pe, perform_fit=do_tau_fit
    )

    if aggregated['delta_t']:
        print("Generating aggregated correlation map...")
        agg_df = pd.DataFrame({
            'delta_t': np.concatenate(aggregated['delta_t']),
            'total_pe': np.concatenate(aggregated['total_pe']),
            'multiplicity': np.concatenate(aggregated['multiplicity'])
        })
        plot_correlation_maps(agg_df, output_dir, agg_label, M1_or_M2)

    if aggregated['low_light_areas']:
        print("Performing aggregated low-light analysis...")
        all_ll_areas = np.concatenate(aggregated['low_light_areas'], axis=0)
        fit_and_plot_low_light(all_ll_areas, output_dir, agg_label, M1_or_M2, hist_range=low_light_fit_range)

    # --- Aggregated SiPM Analysis ---
    if aggregated['sipm_events']:
        print("Generating aggregated SiPM histograms...")
        agg_sipm_df = pd.concat(aggregated['sipm_events'], ignore_index=True)
        plot_sipm_histograms(agg_sipm_df, output_dir, agg_label, M1_or_M2, **sipm_hist_config)

    print("--- Analysis Complete ---")

if __name__ == '__main__':
    main()