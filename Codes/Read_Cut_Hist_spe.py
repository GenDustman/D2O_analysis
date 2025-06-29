#!/usr/bin/env python3
"""
Refactored script for processing ROOT files with detailed configuration
and an additional per-event time-std cut. Modular functions handle I/O,
histogram plotting, Δt computation, and aggregated τ fitting.
Includes low-light (triggerbit=16) analysis with multi-Gaussian fitting.

PERFORMANCE OPTIMIZATIONS:
- Replaced slow pandas row-by-row iteration (iterrows) with vectorized 
  operations using the Awkward Array library for a significant speedup.
- Per-event calculations (like time standard deviation) are now performed 
  efficiently on entire data chunks before being loaded into pandas.
- Increased the Uproot iteration step size to reduce chunking overhead.
- Cleaned up DataFrame creation to avoid storing large, unnecessary list-like
  objects in columns.
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
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


def save_cut_histograms(events, delta_t_range, area_range, bins,
                        save_dir, run_label, time_std_cut, logscale=True):
    """
    Apply sequential cuts on Δt, sum_area, and pre-calculated time-std, then
    save errorbar histograms of Δt and sum_area and pickle their data.

    Args:
        events: DataFrame from compute_delta_t with 'delta_t', 'sum_area', and 'time_std'.
        delta_t_range: Tuple (min_ns, max_ns) for Δt cut.
        area_range: Tuple (min_ADC, max_ADC) for sum_area cut.
        bins: Number of bins for histograms.
        save_dir: Directory to store output files.
        run_label: Text label (e.g., 'Run 123') for legends.
        time_std_cut: Maximum allowed standard deviation of channel times in ns.
        logscale: Whether to use logarithmic y-axis.

    Returns:
        Tuple of numpy arrays (delta_t_vals, sum_area_vals) after all cuts,
        or (None, None) if no events survive.
    """
    dt_min, dt_max = delta_t_range
    s_min, s_max = area_range

    ensure_dir(save_dir)
    sel = events.dropna(subset=['delta_t']).copy()
    print(f"{run_label}: after Δt NaN drop: {len(sel)} events")
    sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
    print(f"{run_label}: after Δt cut: {len(sel)} events")
    sel = sel[(sel['sum_area'] >= s_min) & (sel['sum_area'] <= s_max)]
    print(f"{run_label}: after sum_area cut: {len(sel)} events")
    
    # OPTIMIZED: The slow, row-by-row standard deviation calculation has been removed.
    # The 'time_std' was pre-calculated efficiently in `process_run`.
    # We now apply a simple and fast boolean cut on the existing column.
    sel = sel.dropna(subset=['time_std'])
    sel = sel[sel['time_std'] < time_std_cut]
    print(f"{run_label}: after time-std < {time_std_cut} ns cut: {len(sel)} events")

    if sel.empty:
        return None, None
    dt_bins = np.linspace(dt_min, dt_max, bins+1)
    dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
    dt_err = np.sqrt(dt_counts)
    save_pickle({'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err},
                save_dir / 'delta_t_hist.pkl')
    plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
    plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title('Δt Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_dir / 'delta_t_hist.png'); plt.close()
    s_bins = np.linspace(s_min, s_max, bins+1)
    s_counts, s_edges = np.histogram(sel['sum_area'], bins=s_bins)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    s_err = np.sqrt(s_counts)
    save_pickle({'hist': s_counts, 'centers': s_centers, 'errors': s_err},
                save_dir / 'sum_area_hist.pkl')
    plt.errorbar(s_centers, s_counts, yerr=s_err, fmt='o', label=run_label)
    plt.xlabel('Sum Area (ADC)'); plt.ylabel('Counts'); plt.title('Total Charge Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(save_dir / 'sum_area_hist.png'); plt.close()
    return sel['delta_t'].values, sel['sum_area'].values


def fit_and_plot_low_light(area_data, output_dir, file_label, hist_range, hist_bins=200):
    """
    Plots and fits sum_area for channels 0-11 for low-light events (triggerbit=16).

    Args:
        area_data: Numpy array of shape (n_events, n_channels) with area values.
        output_dir: Directory to save the plot.
        file_label: String to include in the output filename (e.g., 'Run123' or 'Aggregated').
        hist_range: Tuple (min, max) for the area histogram range.
        hist_bins: Number of bins for the area histograms.
    """
    if area_data.size == 0:
        print(f"No low-light data to process for {file_label}.")
        return

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

    for i in range(12):
        ax = axes[i]
        ch_data = area_data[:, i]
        counts, edges = np.histogram(ch_data, bins=hist_bins, range=hist_range)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.hist(ch_data, bins=edges, alpha=0.7, label=f'Ch {i} Data')

        # Initial guesses for the fit
        p0 = [
            counts.max(), 0, 20,          # A0, mu0, sig0 (pedestal)
            counts.max()/5, 100, 30,       # A1, mu1, sig1 (1PE)
            counts.max()/25, counts.max()/125  # A2 (2PE), A3 (3PE)
        ]
        try:
            # Fit only where there are counts to avoid issues
            mask = counts > 0
            popt, pcov = curve_fit(constrained_gaussians, centers[mask], counts[mask], p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

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

        except RuntimeError:
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


def process_run(run, data_dir, output_dir, delta_t_cut, area_cut, bins,
                mult_adc, multiplicity_cut, time_std_cut, logscale, low_light_fit_range):
    """
    Process a single run: read data, perform vectorized calculations,
    make histograms, and apply cuts.

    Args:
        run: Integer run number.
        data_dir: Directory containing ROOT files.
        output_dir: Base directory for outputs.
        delta_t_cut: Δt cut tuple (min, max).
        area_cut: sum_area cut tuple (min, max).
        bins: Number of bins for histograms.
        mult_adc: ADC threshold for multiplicity.
        multiplicity_cut: Multiplicity count threshold.
        time_std_cut: Time-std cut in ns.
        logscale: Log scale for plot y-axes.
        low_light_fit_range: Tuple (min, max) for low-light ADC fit.

    Returns:
        Tuple (delta_t_vals, sum_area_vals, low_light_area_data) or None.
    """
    print(f"Processing run {run}")
    infile = data_dir / f"run{run}_processed_v5.root"
    if not infile.exists():
        print(f"Missing file: {infile}")
        return None
    
    dfs = []
    branches = ['eventID', 'nsTime', 'triggerBits', 'area', 'peakPosition']
    # OPTIMIZED: Increased step_size to reduce the number of chunks and the
    # overhead from pd.concat.
    for chunk in uproot.open(infile)['tree'].iterate(
        branches, library='ak', step_size='500 MB'):
        
        # --- OPTIMIZED: Vectorized Calculations with Awkward Array ---
        areas_ak = chunk['area']
        times_ak = chunk['peakPosition']

        # Work with the first 12 channels for main analysis
        areas_12ch_ak = areas_ak[:, :12]
        times_12ch_ak = times_ak[:, :12]

        # Create a boolean mask for channels passing the ADC cut
        postmcut_mask = areas_12ch_ak > mult_adc

        # --- FIX: Use ak.mask to prevent array flattening and correctly calculate std ---
        # The original boolean array indexing (array[mask]) could flatten the data
        # if a chunk from the ROOT file contained only events with the same number
        # of channels (making it a "regular" array). This flattening caused
        # ak.std(..., axis=1) to fail with a ValueError.
        #
        # The fix is to use ak.mask, which preserves the jagged structure by
        # replacing values that do not satisfy the mask with `None`. The subsequent
        # ak.std call correctly ignores these `None` values.
        masked_times = ak.mask(times_12ch_ak, postmcut_mask)

        # Calculate the standard deviation along axis=1 (per-event).
        # ak.std handles events with 0 or 1 valid hits correctly, returning None or 0.0.
        time_std = ak.std(masked_times, axis=1)

        # --- Create a clean Pandas DataFrame ---
        df = pd.DataFrame({
            'eventID': ak.to_numpy(chunk['eventID']),
            'nsTime': ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            'sum_area': ak.to_numpy(ak.sum(areas_12ch_ak, axis=1)),
            'multiplicity': ak.to_numpy(ak.sum(postmcut_mask, axis=1)),
            'time_std': ak.to_numpy(time_std),
            # Keep 'area_array' for the low-light analysis, converting to list
            'area_array': ak.to_list(areas_ak),
        })
        dfs.append(df)
        
    if not dfs:
        return None
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_pickle(output_dir / f"run{run}_data.pkl")
    
    hist_dir = output_dir / f"run{run}" / "histograms"
    cut_dir = output_dir / f"run{run}" / "cuthist"
    ll_dir = output_dir / f"run{run}" / "lowlight"
    ensure_dir(hist_dir); ensure_dir(cut_dir); ensure_dir(ll_dir)
    
    plot_histogram([df_all['triggerBits'].to_numpy()], ['triggerBits'],
                   np.arange(0, 36), hist_dir / f"{run}_triggerBits.png",
                   'Trigger Bits Distribution', 'triggerBits', logscale)
    plot_histogram([df_all['sum_area'], df_all.loc[df_all['triggerBits'] == 2, 'sum_area']],
                   ['All', 'Trig=2'], np.linspace(0, 100000, bins + 1),
                   hist_dir / f"{run}_sum_area.png", 'Sum Area Comparison', 'ADC', logscale)

    # Low-light (triggerbit=16) analysis
    ll_events = df_all[df_all['triggerBits'] == 16]
    low_light_area_data = np.array(ll_events['area_array'].tolist())[:, :12] if not ll_events.empty else np.array([])
    if low_light_area_data.size > 0:
        fit_and_plot_low_light(low_light_area_data, ll_dir, f'Run{run}', hist_range=low_light_fit_range)
    else:
        print(f"No low-light events found for run {run}. Skipping low-light analysis.")
    
    events = compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)
    
    cut_results = save_cut_histograms(
        events, delta_t_cut, area_cut, bins, cut_dir,
        f"Run {run}", time_std_cut, logscale
    )
    if cut_results:
        dt_vals, sa_vals = cut_results
        return dt_vals, sa_vals, low_light_area_data
    else:
        return None, None, low_light_area_data


def aggregate_plots(aggregated, delta_t_cut, area_cut, bins,
                    fit_window, output_dir, logscale_dt, logscale_sa,
                    perform_fit=True):
    """
    Generate aggregated Δt and sum_area histograms, with an option for a τ fit.

    Args:
        aggregated: Dict with 'delta_t' and 'sum_area_cut' lists.
        delta_t_cut: (min, max) ns.
        area_cut: (min, max) ADC.
        bins: Bin count.
        fit_window: (t_low, t_high) ns for fitting τ.
        output_dir: Directory for aggregated outputs.
        logscale_dt: Use log y-axis for the Δt plot.
        logscale_sa: Use log y-axis for the sum_area plot.
        perform_fit: If True, perform and plot an exponential fit on the Δt data.
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

    all_sa = np.concatenate(aggregated['sum_area_cut']) if aggregated['sum_area_cut'] else np.array([])
    if all_sa.size:
        sa_min, sa_max = area_cut
        sa_bins = np.linspace(sa_min, sa_max, bins + 1)
        hist_sa, sa_edges = np.histogram(all_sa, bins=sa_bins)
        sa_centers = 0.5 * (sa_edges[:-1] + sa_edges[1:])
        sa_err = np.sqrt(hist_sa)
        plt.errorbar(sa_centers, hist_sa, yerr=sa_err, fmt='o', label=f'Runs')
        plt.xlabel('Total Charge (ADC)'); plt.ylabel('Counts'); plt.title(f'Aggregated Total Charge')
        if logscale_sa: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        plt.savefig(output_dir / 'aggregated_sum_area.png'); plt.close()
        save_pickle({'centers': sa_centers, 'hist': hist_sa, 'errors': sa_err},
                    output_dir / 'aggregated_sum_area.pkl')


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
    delta_t_cut         = (0, 10000)      # Δt range in ns
    area_cut            = (0, 50000)      # Total charge (ADC) range
    bins                = 20             # Bins for cut histograms
    multiplicity_adc    = 2*100           # ADC threshold per channel
    multiplicity_cut    = 2               # Min channels above threshold
    time_std_cut        = 2.5*16          # Max std of channel times in ns
    logscale            = True            # Use log scale for y-axes
    logscale_dt         = True            # Use log scale for Δt histogram
    logscale_sa         = False           # Use log scale for sum_area histogram
    do_tau_fit          = False            # Whether to perform exponential fit on Δt
    tau_fit_window      = (2500, 10000)   # Fit window for τ in ns
    low_light_fit_range = (-50, 400)      # Fit window for low-light analysis in ADC
    # --------------------------------
    dt_min, dt_max = delta_t_cut
    sa_min, sa_max = area_cut
    print("=== Configuration ===")
    print(f"Runs: {start_run} to {end_run}")
    print(f"Δt cut: {delta_t_cut}")
    print(f"Area cut: {area_cut}")
    print(f"Bins: {bins}")
    print(f"Multiplicity ADC threshold: {multiplicity_adc}")
    print(f"Multiplicity cut: {multiplicity_cut}")
    print(f"Time-std cut: < {time_std_cut} ns")
    print(f"Perform τ fit: {do_tau_fit}")
    if do_tau_fit:
        print(f"τ fit window: {tau_fit_window} ns")
    print(f"Low-light fit range: {low_light_fit_range} ADC")
    print(f"Logscale: {logscale}")
    print("======================")

    data_dir = Path('/raid1/genli/Data_D2O')
    output_dir = data_dir / (
        f"runs_{start_run}_{end_run}_dt{dt_min}-{dt_max}"
        f"_sa{sa_min}-{sa_max}_madc{multiplicity_adc}_mcut{multiplicity_cut}_std{time_std_cut}"
    )
    ensure_dir(output_dir)

    aggregated = {
        'delta_t': [],
        'sum_area_cut': [],
        'low_light_areas': []
    }

    for run in range(start_run, end_run + 1):
        result = process_run(
            run,
            data_dir,
            output_dir,
            delta_t_cut,
            area_cut,
            bins,
            multiplicity_adc,
            multiplicity_cut,
            time_std_cut,
            logscale,
            low_light_fit_range
        )
        if result:
            dt_vals, sa_vals, ll_areas = result
            if dt_vals is not None:
                aggregated['delta_t'].append(dt_vals)
            if sa_vals is not None:
                aggregated['sum_area_cut'].append(sa_vals)
            if ll_areas.size > 0:
                aggregated['low_light_areas'].append(ll_areas)

    aggregate_plots(
        aggregated,
        delta_t_cut,
        area_cut,
        bins,
        tau_fit_window,
        output_dir,
        logscale_dt,
        logscale_sa,
        perform_fit=do_tau_fit
    )

    if aggregated['low_light_areas']:
        print("Performing aggregated low-light analysis...")
        all_ll_areas = np.concatenate(aggregated['low_light_areas'], axis=0)
        print('plotting aggregated low-light fits...')
        fit_and_plot_low_light(all_ll_areas, output_dir, 'Aggregated', hist_range=low_light_fit_range)

if __name__ == '__main__':
    main()
