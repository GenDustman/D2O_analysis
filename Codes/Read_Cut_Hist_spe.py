#!/usr/bin/env python3
"""
Refactored script for processing ROOT files with detailed configuration
and an additional per-event time-std cut. Modular functions handle I/O,
histogram plotting, Δt computation, and aggregated τ fitting.
Includes low-light (triggerbit=16) analysis with multi-Gaussian fitting
and new 3x3 correlation maps for key variables with correlation coefficients.

MODIFICATION:
- Changed analysis from total charge (ADC) to total photoelectrons (P.E.).
- The conversion is done by dividing each channel's area by its fitted single
  photoelectron mean (mu1) from low-light data.
- Error propagation for the P.E. histograms now correctly includes the
  uncertainty from the mu1 parameter fit, combined in quadrature with the
  Poisson error of the bin counts.
"""
import sys
from pathlib import Path
import pickle
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

    Args:
        path: pathlib.Path of the directory to ensure exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(data: dict, path: Path):
    """
    Serialize and save a Python dictionary to a pickle file.

    Args:
        data: Dictionary of data to pickle.
        path: Path where the pickle file will be written.
    """
    with path.open('wb') as f:
        pickle.dump(data, f)


def plot_histogram(arrays, labels, bins, img_path, title, xlabel,
                   logscale=True, figsize=(10, 6)):
    """
    Plot one or more datasets as overlapping histograms, with consistent styling.

    Args:
        arrays: List of 1D numpy arrays of values to histogram.
        labels: Corresponding list of legend labels.
        bins: Bin edges or count for histogram.
        img_path: Path to save the histogram PNG.
        title: Plot title.
        xlabel: Label for the x-axis.
        logscale: If True, use logarithmic y-axis.
        figsize: Figure size tuple.

    Returns:
        List of tuples (counts, edges) for each array.
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
    plt.title(title)
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


def plot_correlation_maps(df, output_dir, label):
    """
    Plots a 3x3 grid of correlation maps for delta_t, total_pe, and multiplicity.
    Diagonal plots are 1D histograms. Off-diagonal are 2D histograms with the
    Pearson correlation coefficient displayed.

    Args:
        df: DataFrame with 'delta_t', 'total_pe', and 'multiplicity' columns.
        output_dir: Directory to save the plot.
        label: Label for the plot title (e.g., 'Run 123' or 'Aggregated').
    """
    ensure_dir(output_dir)
    if df.empty:
        print(f"DataFrame is empty for {label}. Skipping correlation map.")
        return

    variables = ['delta_t', 'total_pe', 'multiplicity']
    pretty_labels = ['Δt (ns)', 'Total Photoelectrons', 'Multiplicity']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Correlation Matrix ({label})', fontsize=18)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            var_y = variables[i]
            var_x = variables[j]

            # Set axis labels for the outer plots only
            if i == 2:  # Bottom row
                ax.set_xlabel(pretty_labels[j], fontsize=12)
            if j == 0:  # Leftmost column
                ax.set_ylabel(pretty_labels[i], fontsize=12)

            # Diagonal plots: 1D histograms
            if i == j:
                data = df[var_x].dropna()
                if not data.empty:
                    ax.hist(data, bins=50, histtype='step', linewidth=1.5, color='k')
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle=':')
            # Off-diagonal plots: 2D histograms
            else:
                subset = df[[var_x, var_y]].dropna()
                # Pearson correlation requires at least 2 data points
                if not subset.empty and len(subset) > 1:
                    h = ax.hist2d(subset[var_x], subset[var_y],
                                  bins=50, cmap='viridis', norm=LogNorm())
                    if h[0].max() > 0:
                        fig.colorbar(h[3], ax=ax)

                    # --- Calculate and display Pearson correlation ---
                    corr, _ = pearsonr(subset[var_x], subset[var_y])
                    corr_text = f'Corr: {corr:.2f}'
                    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

            # Remove tick labels for inner plots to reduce clutter
            if i < 2:
                ax.tick_params(axis='x', labelbottom=False)
            if j > 0:
                ax.tick_params(axis='y', labelleft=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename_label = label.replace(" ", "_").replace(":", "")
    save_path = output_dir / f'{filename_label}_correlation_map.png'
    plt.savefig(save_path)
    print(f"Correlation map saved to {save_path}")
    plt.close()


def compute_delta_t(df, muon_bits, veto_bits, mult_thresh):
    """
    Compute time differences Δt between veto events and the preceding muon event.
    This function is already efficient and does not require changes.

    Args:
        df: DataFrame containing 'nsTime' and 'triggerBits'.
        muon_bits: Minimum triggerBits value to classify as a muon event.
        veto_bits: Exact triggerBits value for veto events of interest.
        mult_thresh: Minimum multiplicity for a veto event to be considered.

    Returns:
        DataFrame of veto events with a new 'delta_t' column in ns.
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
                        save_dir, run_label, time_std_cut, logscale=True):
    """
    Apply sequential cuts on Δt, total_pe, and pre-calculated time-std, then
    save errorbar histograms of Δt and total_pe and pickle their data.
    Also generates and saves a 3x3 correlation map of the final cut data.

    Args:
        events: DataFrame with 'delta_t', 'total_pe', 'total_pe_err', and 'time_std'.
        delta_t_range: Tuple (min_ns, max_ns) for Δt cut.
        pe_range: Tuple (min_pe, max_pe) for total_pe cut.
        bins: Number of bins for histograms.
        save_dir: Directory to store output files.
        run_label: Text label (e.g., 'Run 123') for legends.
        time_std_cut: Maximum allowed standard deviation of channel times in ns.
        logscale: Whether to use logarithmic y-axis.

    Returns:
        Tuple of numpy arrays (delta_t_vals, total_pe_vals, multiplicity_vals, total_pe_err_vals)
        after all cuts, or (None, None, None, None) if no events survive.
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

    # Plot correlation maps for the single run after all cuts
    plot_correlation_maps(sel, save_dir, run_label)

    if sel.empty:
        return None, None, None, None

    # --- Delta T Histogram ---
    dt_bins = np.linspace(dt_min, dt_max, bins + 1)
    dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
    dt_err = np.sqrt(dt_counts) # Simple Poisson error
    save_pickle({'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err},
                save_dir / 'delta_t_hist.pkl')
    plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
    plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title('Δt Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_dir / 'delta_t_hist.png'); plt.close()

    # --- Total PE Histogram with Propagated Error ---
    pe_bins = np.linspace(pe_min, pe_max, bins + 1)
    pe_counts, pe_edges = np.histogram(sel['total_pe'], bins=pe_bins)
    pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])

    # Calculate the propagated uncertainty from the gain fit for each bin
    # This is the sum in quadrature of individual event uncertainties in that bin
    gain_err_sq, _ = np.histogram(sel['total_pe'], bins=pe_bins, weights=sel['total_pe_err']**2)
    
    # Total error is the combination of Poisson counting error and gain error
    total_err = np.sqrt(pe_counts + gain_err_sq)

    save_pickle({'hist': pe_counts, 'centers': pe_centers, 'errors': total_err},
                save_dir / 'total_pe_hist.pkl')
    plt.errorbar(pe_centers, pe_counts, yerr=total_err, fmt='o', label=run_label)
    plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title('Total Photoelectron Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_dir / 'total_pe_hist.png'); plt.close()

    return sel['delta_t'].values, sel['total_pe'].values, sel['multiplicity'].values, sel['total_pe_err'].values


def fit_and_plot_low_light(area_data, output_dir, file_label, hist_range, hist_bins=200):
    """
    Plots and fits sum_area for channels 0-11 for low-light events (triggerbit=16).

    Args:
        area_data: Numpy array of shape (n_events, n_channels) with area values.
        output_dir: Directory to save the plot.
        file_label: String to include in the output filename (e.g., 'Run123' or 'Aggregated').
        hist_range: Tuple (min, max) for the area histogram range.
        hist_bins: Number of bins for the area histograms.

    Returns:
        Tuple of (mu1_values, mu1_errors), where each is a numpy array of shape (12,).
        Returns NaNs for channels where the fit failed.
    """
    if area_data.size == 0:
        print(f"No low-light data to process for {file_label}.")
        return np.full(12, np.nan), np.full(12, np.nan)

    def constrained_gaussians(x, a0, mu0, sig0, a1, mu1, sig1, a2, a3):
        """
        Fit function: pedestal + 3 constrained photoelectron peaks.
        """
        # Ensure variance is non-negative
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
    fig.suptitle(f'Low-Light Channel Area Fits ({file_label})', fontsize=16)
    axes = axes.flatten()
    
    mu1_values = np.full(12, np.nan)
    mu1_errors = np.full(12, np.nan)

    for i in range(12):
        ax = axes[i]
        ch_data = area_data[:, i]
        counts, edges = np.histogram(ch_data, bins=hist_bins, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.hist(ch_data, bins=edges, alpha=0.7, label=f'Ch {i} Data')

        # Initial guesses for the fit
        p0 = [
            counts.max(), 0, 20,           # A0, mu0, sig0 (pedestal)
            counts.max()/5, 100, 30,       # A1, mu1, sig1 (1PE)
            counts.max()/25, counts.max()/125  # A2 (2PE), A3 (3PE)
        ]
        try:
            # Fit only where there are counts to avoid issues
            mask = counts > 0
            popt, pcov = curve_fit(constrained_gaussians, centers[mask], counts[mask], p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            # Store the fitted mu1 and its error for this channel
            mu1_values[i] = popt[4]
            mu1_errors[i] = perr[4]

            fit_x = np.linspace(hist_range[0], hist_range[1], 500)
            ax.plot(fit_x, constrained_gaussians(fit_x, *popt), 'r-', label='Fit')

            # Create legend with fit parameters
            param_text = (
                f'$\\mu_0$: {popt[1]:.1f} ± {perr[1]:.1f}\n'
                f'$\\sigma_0$: {popt[2]:.1f} ± {perr[2]:.1f}\n'
                f'$\\mu_1$: {popt[4]:.1f} ± {perr[4]:.1f}\n'
                f'$\\sigma_1$: {popt[5]:.1f} ± {perr[5]:.1f}'
            )
            ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except (RuntimeError, ValueError):
            ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, color='red',
                    ha='center', va='center')

        ax.set_title(f'Channel {i}')
        ax.set_xlabel('Sum Area (ADC)')
        ax.set_ylabel('Events')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle=':')
        ax.legend(loc='lower left', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    ensure_dir(output_dir)
    plt.savefig(output_dir / f'{file_label}_low_light_fits.png')
    print(f"Low-light fits saved to {output_dir / f'{file_label}_low_light_fits.png'}")
    plt.close()
    
    return mu1_values, mu1_errors


def calculate_total_pe(df, mu1_values, mu1_errors):
    """
    Calculates the total photoelectrons and its uncertainty for each event.

    Args:
        df (pd.DataFrame): DataFrame containing an 'area_array' column.
        mu1_values (np.ndarray): Array of shape (12,) with mu1 gain for each channel.
        mu1_errors (np.ndarray): Array of shape (12,) with error on mu1 gain.

    Returns:
        tuple[np.ndarray, np.ndarray]: (total_pe, total_pe_err) for each event.
    """
    if np.all(np.isnan(mu1_values)):
        print("ERROR: Low-light fit failed. Cannot calculate photoelectrons.")
        nan_array = np.full(len(df), np.nan)
        return nan_array, nan_array

    # Replace NaN or non-positive values with infinity to make the division result in zero.
    mu1_safe = np.where(np.isnan(mu1_values) | (mu1_values <= 0), np.inf, mu1_values)
    mu1_err_safe = np.where(np.isnan(mu1_errors), 0, mu1_errors) # Treat NaN error as zero

    if np.any(mu1_safe == np.inf):
        nan_ch = np.where(np.isnan(mu1_values) | (mu1_values <= 0))[0]
        print(f"Warning: mu1 fit failed/invalid for channels {nan_ch}. "
              "These channels will be excluded from the P.E. sum.")
    
    area_data_np = np.array(df['area_array'].to_list())[:, :12]
    
    # Calculate PE per channel
    pe_per_channel = area_data_np / mu1_safe
    total_pe = np.sum(pe_per_channel, axis=1)

    # Propagate error: variance_pe = sum( (pe_i * (err_mu_i / mu_i))^2 )
    relative_err_sq = (mu1_err_safe / mu1_safe)**2
    pe_variance = np.sum((pe_per_channel**2) * relative_err_sq, axis=1)
    total_pe_err = np.sqrt(pe_variance)
    
    return total_pe, total_pe_err


def process_run(run, data_dir, output_dir, delta_t_cut, pe_cut, bins,
                mult_adc, multiplicity_cut, time_std_cut, logscale, low_light_fit_range):
    """
    Process a single run: read data, perform vectorized calculations,
    make histograms, and apply cuts.

    Args:
        ...
        pe_cut: total_pe cut tuple (min, max).
        ...

    Returns:
        Tuple (delta_t_vals, total_pe_vals, multiplicity_vals, total_pe_err_vals,
               low_light_area_data, mu1_values_run, mu1_errors_run) or None.
    """
    print(f"Processing run {run}")
    infile = data_dir / f"run{run}_processed_v5.root"
    if not infile.exists():
        print(f"Missing file: {infile}")
        return None

    dfs = []
    branches = ['eventID', 'nsTime', 'triggerBits', 'area', 'peakPosition']
    for chunk in uproot.open(infile)['tree'].iterate(
        branches, library='ak', step_size='500 MB'):

        areas_ak = chunk['area']
        times_ak = chunk['peakPosition']
        areas_12ch_ak = areas_ak[:, :12]
        times_12ch_ak = times_ak[:, :12]
        postmcut_mask = areas_12ch_ak > mult_adc
        masked_times = ak.mask(times_12ch_ak, postmcut_mask)
        time_std = ak.std(masked_times, axis=1)

        df = pd.DataFrame({
            'eventID': ak.to_numpy(chunk['eventID']),
            'nsTime': ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            'multiplicity': ak.to_numpy(ak.sum(postmcut_mask, axis=1)),
            'time_std': ak.to_numpy(time_std),
            'area_array': ak.to_list(areas_ak),
        })
        dfs.append(df)

    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    
    run_dir = output_dir / f"run{run}"
    hist_dir = run_dir / "histograms"
    cut_dir = run_dir / "cuthist"
    ll_dir = run_dir / "lowlight"
    ensure_dir(hist_dir); ensure_dir(cut_dir); ensure_dir(ll_dir)

    plot_histogram([df_all['triggerBits'].to_numpy()], ['triggerBits'],
                   np.arange(0, 36), hist_dir / f"{run}_triggerBits.png",
                   'Trigger Bits Distribution', 'triggerBits', logscale)

    # --- Low-light analysis and P.E. Conversion ---
    ll_events = df_all[df_all['triggerBits'] == 16]
    low_light_area_data = np.array(ll_events['area_array'].to_list())[:, :12] if not ll_events.empty else np.array([])
    
    mu1_values_run = np.full(12, np.nan)
    mu1_errors_run = np.full(12, np.nan)
    if low_light_area_data.size > 0:
        mu1_values_run, mu1_errors_run = fit_and_plot_low_light(low_light_area_data, ll_dir, f'Run{run}', hist_range=low_light_fit_range)
        print(f"Run {run} fitted mu1 values (P.E. gain): {np.round(mu1_values_run, 2)}")
        save_pickle({'mu1_values': mu1_values_run, 'mu1_errors': mu1_errors_run}, ll_dir / f'run{run}_mu1_fits.pkl')
    else:
        print(f"No low-light events found for run {run}. Skipping low-light analysis.")

    # Calculate total P.E. and its uncertainty, adding new columns to the DataFrame
    df_all['total_pe'], df_all['total_pe_err'] = calculate_total_pe(df_all, mu1_values_run, mu1_errors_run)
    df_all.to_pickle(run_dir / f"run{run}_data_with_pe.pkl")

    # Plot preliminary P.E. histogram (replaces sum_area)
    plot_histogram([df_all['total_pe'].dropna(), df_all.loc[df_all['triggerBits'] == 2, 'total_pe'].dropna()],
                   ['All', 'Trig=2'], np.linspace(0, 2000, bins + 1), # Adjusted range for P.E.
                   hist_dir / f"{run}_total_pe.png", 'Total Photoelectron Comparison', 'Total P.E.', logscale)

    # --- Event Selection and Final Plots ---
    events = compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)

    cut_results = save_cut_histograms(
        events, delta_t_cut, pe_cut, bins, cut_dir,
        f"Run {run}", time_std_cut, logscale
    )
    if cut_results:
        dt_vals, pe_vals, mult_vals, pe_err_vals = cut_results
        return dt_vals, pe_vals, mult_vals, pe_err_vals, low_light_area_data, mu1_values_run, mu1_errors_run
    else:
        return None, None, None, None, low_light_area_data, mu1_values_run, mu1_errors_run


def aggregate_plots(aggregated, delta_t_cut, pe_cut, bins,
                    fit_window, output_dir, logscale_dt, logscale_pe,
                    perform_fit=True):
    """
    Generate aggregated Δt and total_pe histograms, with an option for a τ fit.

    Args:
        aggregated: Dict with 'delta_t', 'total_pe', and 'total_pe_err' lists.
        delta_t_cut: (min, max) ns.
        pe_cut: (min, max) P.E.
        ...
        logscale_pe: Use log y-axis for the total_pe plot.
        ...
    """
    ensure_dir(output_dir)
    dt_min, dt_max = delta_t_cut
    all_dt = np.concatenate(aggregated['delta_t']) if aggregated['delta_t'] else np.array([])

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

        plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'Aggregated Δt')
        if logscale_dt: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_delta_t.png'); plt.close()
        save_pickle(pickle_data, output_dir / 'aggregated_delta_t.pkl')

    # --- Aggregated P.E. Plot with Propagated Error ---
    all_pe = np.concatenate(aggregated['total_pe']) if aggregated['total_pe'] else np.array([])
    all_pe_err = np.concatenate(aggregated['total_pe_err']) if aggregated['total_pe_err'] else np.array([])
    
    if all_pe.size:
        pe_min, pe_max = pe_cut
        pe_bins = np.linspace(pe_min, pe_max, bins + 1)
        hist_pe, pe_edges = np.histogram(all_pe, bins=pe_bins)
        pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])

        # Propagate gain uncertainty
        gain_err_sq, _ = np.histogram(all_pe, bins=pe_bins, weights=all_pe_err**2)
        
        # Combine with Poisson uncertainty
        total_err = np.sqrt(hist_pe + gain_err_sq)
        
        plt.errorbar(pe_centers, hist_pe, yerr=total_err, fmt='o', label=f'Runs')
        plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title(f'Aggregated Total Photoelectrons')
        if logscale_pe: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_total_pe.png'); plt.close()
        save_pickle({'centers': pe_centers, 'hist': hist_pe, 'errors': total_err},
                    output_dir / 'aggregated_total_pe.pkl')


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
    delta_t_cut 	= (0, 10000)      # Δt range in ns
    pe_cut 		= (0, 1000)       # Total Photoelectron (P.E.) range
    bins 		= 20              # Bins for cut histograms
    multiplicity_adc 	= 5 * 100         # ADC threshold per channel
    multiplicity_cut 	= 2               # Min channels above threshold
    time_std_cut 	= 2.5 * 16        # Max std of channel times in ns
    logscale 		= True            # Use log scale for y-axes in single runs
    logscale_dt 	= True            # Use log scale for aggregated Δt histogram
    logscale_pe 	= False           # Use log scale for aggregated P.E. histogram
    do_tau_fit 		= True            # Whether to perform exponential fit on Δt
    tau_fit_window 	= (2500, 10000)   # Fit window for τ in ns
    low_light_fit_range = (-50, 400)        # Fit window for low-light analysis in ADC
    # --------------------------------
    dt_min, dt_max = delta_t_cut
    pe_min, pe_max = pe_cut
    print("=== Configuration ===")
    print(f"Runs: {start_run} to {end_run}")
    print(f"Δt cut: {delta_t_cut} ns")
    print(f"Photoelectron cut: {pe_cut} P.E.")
    print(f"Bins: {bins}")
    print(f"Multiplicity ADC threshold: {multiplicity_adc}")
    print(f"Multiplicity cut: {multiplicity_cut}")
    print(f"Time-std cut: < {time_std_cut} ns")
    print(f"Perform τ fit: {do_tau_fit}")
    if do_tau_fit:
        print(f"τ fit window: {tau_fit_window} ns")
    print(f"Low-light fit range: {low_light_fit_range} ADC")
    print("======================")

    data_dir = Path('/raid1/genli/Data_D2O')
    output_dir = data_dir / (
        f"runs_{start_run}_{end_run}_dt{dt_min}-{dt_max}"
        f"_pe{pe_min}-{pe_max}_madc{multiplicity_adc}_mcut{multiplicity_cut}_std{time_std_cut}"
    )
    ensure_dir(output_dir)

    aggregated = {
        'delta_t': [],
        'total_pe': [],
        'total_pe_err': [],
        'multiplicity': [],
        'low_light_areas': [],
        'mu1_values': [],
        'mu1_errors': []
    }

    for run in range(start_run, end_run + 1):
        result = process_run(
            run,
            data_dir,
            output_dir,
            delta_t_cut,
            pe_cut,
            bins,
            multiplicity_adc,
            multiplicity_cut,
            time_std_cut,
            logscale,
            low_light_fit_range
        )
        if result:
            dt_vals, pe_vals, mult_vals, pe_err_vals, ll_areas, mu1s, mu1_errs = result
            if dt_vals is not None:
                aggregated['delta_t'].append(dt_vals)
            if pe_vals is not None:
                aggregated['total_pe'].append(pe_vals)
                aggregated['total_pe_err'].append(pe_err_vals)
            if mult_vals is not None:
                aggregated['multiplicity'].append(mult_vals)
            if ll_areas.size > 0:
                aggregated['low_light_areas'].append(ll_areas)
            if mu1s is not None:
                aggregated['mu1_values'].append(mu1s)
                aggregated['mu1_errors'].append(mu1_errs)

    aggregate_plots(
        aggregated,
        delta_t_cut,
        pe_cut,
        bins,
        tau_fit_window,
        output_dir,
        logscale_dt,
        logscale_pe,
        perform_fit=do_tau_fit
    )

    # Generate and save the aggregated correlation map
    if aggregated['delta_t']:  # Check if there is any data to plot
        print("Generating aggregated correlation map...")
        agg_df = pd.DataFrame({
            'delta_t': np.concatenate(aggregated['delta_t']),
            'total_pe': np.concatenate(aggregated['total_pe']),
            'multiplicity': np.concatenate(aggregated['multiplicity'])
        })
        plot_correlation_maps(agg_df, output_dir, "Aggregated")

    if aggregated['low_light_areas']:
        print("Performing aggregated low-light analysis...")
        all_ll_areas = np.concatenate(aggregated['low_light_areas'], axis=0)
        print('Plotting aggregated low-light fits...')
        agg_mu1s, agg_mu1_errs = fit_and_plot_low_light(all_ll_areas, output_dir, 'Aggregated', hist_range=low_light_fit_range)
        print(f"Aggregated fitted mu1 values (P.E. gain): {np.round(agg_mu1s, 2)}")
        save_pickle({
            'mu1_values_aggregated_fit': agg_mu1s,
            'mu1_errors_aggregated_fit': agg_mu1_errs,
            'mu1_values_per_run': aggregated['mu1_values'],
            'mu1_errors_per_run': aggregated['mu1_errors']
        }, output_dir / 'aggregated_mu1_fits.pkl')


if __name__ == '__main__':
    main()