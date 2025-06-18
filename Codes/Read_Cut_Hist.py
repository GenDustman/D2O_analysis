#!/usr/bin/env python3
"""
Refactored script for processing ROOT files with detailed configuration
and an additional per-event time-std cut. Modular functions handle I/O,
histogram plotting, Δt computation, and aggregated τ fitting.
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    # Draw each histogram with semi-transparent bars
    for data, label in zip(arrays, labels):
        counts, edges, _ = plt.hist(
            data,
            bins=bins,
            alpha=0.7,
            edgecolor='black',
            label=label
        )
        outputs.append((counts, edges))
    # Set axes labels and title
    plt.xlabel(xlabel)
    plt.ylabel('Events')
    plt.title(title)
    # Optionally set log scale
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.minorticks_on()
    # Add grid lines for readability
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    # Save and close figure
    plt.savefig(img_path)
    plt.close()
    return outputs


def compute_delta_t(df, muon_bits, veto_bits, mult_thresh):
    """
    Compute time differences Δt between veto events and the preceding muon event.

    Args:
        df: DataFrame containing 'nsTime' and 'triggerBits'.
        muon_bits: Minimum triggerBits value to classify as a muon event.
        veto_bits: Exact triggerBits value for veto events of interest.
        mult_thresh: Minimum multiplicity for a veto event to be considered.

    Returns:
        DataFrame of veto events with a new 'delta_t' column in ns.
    """
    # Identify muon events (triggerBits >= threshold)
    muon_mask = df['triggerBits'] >= muon_bits
    # Identify veto events (exact triggerBits AND multiplicity requirement)
    veto_mask = (df['triggerBits'] == veto_bits) & (df['multiplicity'] > mult_thresh)
    muon_times = df.loc[muon_mask, 'nsTime'].values
    # Copy the veto events to avoid SettingWithCopy
    events = df.loc[veto_mask].copy()
    times = events['nsTime'].values
    # Find insertion indices: position in muon_times just after each veto time
    idx = np.searchsorted(muon_times, times, side='right')
    delta_t = np.full(times.shape, np.nan)
    valid = idx > 0
    # Compute Δt only where a preceding muon exists
    delta_t[valid] = times[valid] - muon_times[idx[valid] - 1]
    events['delta_t'] = delta_t
    return events


def save_cut_histograms(events, delta_t_range, area_range, bins,
                        save_dir, run_label, time_std_cut, logscale=True):
    """
    Apply sequential cuts on Δt, sum_area, and per-event time-std, then
    save errorbar histograms of Δt and sum_area and pickle their data.

    Args:
        events: DataFrame from compute_delta_t with 'delta_t' and 'sum_area'.
        delta_t_range: Tuple (min_ns, max_ns) for Δt cut.
        area_range: Tuple (min_ADC, max_ADC) for sum_area cut.
        bins: Number of bins for histograms.
        save_dir: Directory to store output files.
        run_label: Text label (e.g., 'Run 123') for legends.
        time_std_cut: Maximum allowed standard deviation of channels 0–11 in ns.
        logscale: Whether to use logarithmic y-axis.

    Returns:
        Tuple of numpy arrays (delta_t_vals, sum_area_vals) after all cuts,
        or (None, None) if no events survive.
    """
    dt_min, dt_max = delta_t_range
    s_min, s_max = area_range

    ensure_dir(save_dir)
    # Drop events without a computed Δt
    sel = events.dropna(subset=['delta_t']).copy()
    print(f"{run_label}: after Δt NaN drop: {len(sel)} events")

    # Δt cut
    sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
    print(f"{run_label}: after Δt cut: {len(sel)} events")

    # sum_area cut
    sel = sel[(sel['sum_area'] >= s_min) & (sel['sum_area'] <= s_max)]
    print(f"{run_label}: after sum_area cut: {len(sel)} events")

    # Compute per-event time-std over channels 0–11 from 'time_array'
    std_vals = np.array([np.std(arr[:12]) for arr in sel['time_array']])
    # time-std cut
    sel = sel[std_vals < time_std_cut]
    print(f"{run_label}: after time-std < {time_std_cut} ns cut: {len(sel)} events")

    if sel.empty:
        return None, None

    # Prepare Δt histogram
    dt_bins = np.linspace(dt_min, dt_max, bins+1)
    dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
    dt_err = np.sqrt(dt_counts)
    # Save histogram data
    save_pickle(
        {'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err},
        save_dir / 'delta_t_hist.pkl'
    )
    # Plot Δt errorbar
    plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
    plt.xlabel('Δt (ns)')
    plt.ylabel('Counts')
    plt.title('Δt Histogram')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'delta_t_hist.png')
    plt.close()

    # Prepare sum_area histogram
    s_bins = np.linspace(s_min, s_max, bins+1)
    s_counts, s_edges = np.histogram(sel['sum_area'], bins=s_bins)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    s_err = np.sqrt(s_counts)
    save_pickle(
        {'hist': s_counts, 'centers': s_centers, 'errors': s_err},
        save_dir / 'sum_area_hist.pkl'
    )
    # Plot sum_area errorbar
    plt.errorbar(s_centers, s_counts, yerr=s_err, fmt='o', label=run_label)
    plt.xlabel('Sum Area (ADC)')
    plt.ylabel('Counts')
    plt.title('Total Charge Histogram')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'sum_area_hist.png')
    plt.close()

    # Return arrays for aggregation
    return sel['delta_t'].values, sel['sum_area'].values


def process_run(run, data_dir, output_dir, delta_t_cut, area_cut, bins,
                mult_adc, mult_thresh, time_std_cut, logscale):
    """
    Process a single run: read data, make histograms, apply cuts.

    Args:
        run: Integer run number.
        data_dir: Directory containing ROOT files.
        output_dir: Base directory for outputs.
        delta_t_cut: Δt cut tuple (min, max).
        area_cut: sum_area cut tuple (min, max).
        bins: Number of bins for histograms.
        mult_adc: ADC threshold for multiplicity.
        mult_thresh: Multiplicity count threshold.
        time_std_cut: Time-std cut in ns.
        logscale: Log scale for plot y-axes.

    Returns:
        Tuple (delta_t_vals, sum_area_vals) or None.
    """
    print(f"Processing run {run}")
    infile = data_dir / f"run{run}_processed_v5.root"
    if not infile.exists():
        print(f"Missing file: {infile}")
        return None

    # Iterate tree in chunks to build full DataFrame
    dfs = []
    for chunk in uproot.open(infile)['tree'].iterate(
        ['eventID','nsTime','triggerBits','area','pulseH'], library='ak', step_size='100 MB'):
        areas = ak.to_numpy(chunk['area'])
        times_ch = ak.to_numpy(chunk['pulseH'])  # channel times
        df = pd.DataFrame({
            'eventID': ak.to_numpy(chunk['eventID']),
            'nsTime': ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            # sum of first 12 channels
            'sum_area': np.sum(areas[:, :12], axis=1),
            # count of channels above ADC threshold
            'multiplicity': np.sum(areas[:, :12] > mult_adc, axis=1),
            'time_array': list(times_ch)
        })
        dfs.append(df)
    if not dfs:
        return None
    # Concatenate all chunks and pickle raw DataFrame
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_pickle(output_dir / f"run{run}_data.pkl")

    # Prepare output dirs
    hist_dir = output_dir / f"run{run}" / "histograms"
    cut_dir = output_dir / f"run{run}" / "cuthist"
    ensure_dir(hist_dir)
    ensure_dir(cut_dir)

    # Basic histograms for triggerBits and sum_area
    plot_histogram(
        [df_all['triggerBits'].to_numpy()], ['triggerBits'],
        np.arange(0, 36),
        hist_dir / f"{run}_triggerBits.png",
        'Trigger Bits Distribution',
        'triggerBits',
        logscale
    )
    plot_histogram(
        [df_all['sum_area'], df_all.loc[df_all['triggerBits']==2, 'sum_area']],
        ['All', 'Trig=2'],
        np.linspace(0, 100000, bins+1),
        hist_dir / f"{run}_sum_area.png",
        'Sum Area Comparison',
        'ADC',
        logscale
    )

    # Compute Δt and apply cuts
    events = compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=mult_thresh)
    return save_cut_histograms(
        events,
        delta_t_cut,
        area_cut,
        bins,
        cut_dir,
        f"Run {run}",
        time_std_cut,
        logscale
    )


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
    delta_t_cut      = (0, 20000)    # Δt range in ns
    area_cut         = (0, 200000)   # Total charge (ADC) range
    bins             = 20            # Bins for cut histograms
    multiplicity_adc = 2*100         # ADC threshold per channel
    multiplicity_cut = 2             # Min channels above threshold
    time_std_cut     = 2.5*16        # Max std of channel times in ns, 1 ADC = 16 ns
    logscale         = True          # Use log scale for y-axes
    # --------------------------------
    
    dt_min, dt_max = delta_t_cut
    sa_min, sa_max = area_cut
    
    # Print overall configuration
    print("=== Configuration ===")
    print(f"Runs: {start_run} to {end_run}")
    print(f"Δt cut: {delta_t_cut}")
    print(f"Area cut: {area_cut}")
    print(f"Bins: {bins}")
    print(f"Multiplicity ADC threshold: {multiplicity_adc}")
    print(f"Multiplicity cut: {multiplicity_cut}")
    print(f"Time-std cut: < {time_std_cut} ns")
    print(f"Logscale: {logscale}")
    print("======================")

    data_dir = Path('/raid1/genli/Data_D2O')
    output_dir = data_dir / (
        f"runs_{start_run}_{end_run}_dt{dt_min}-{dt_max}"
        f"_sa{sa_min}-{sa_max}_mcut{multiplicity_cut}_std{time_std_cut}"
    )
    ensure_dir(output_dir)

    # Containers for aggregated data
    aggregated = {
        'delta_t': [],
        'sum_area_cut': []
    }

    # Loop through runs and process
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
            logscale
        )
        if result:
            dt_vals, sa_vals = result
            aggregated['delta_t'].append(dt_vals)
            aggregated['sum_area_cut'].append(sa_vals)

    # --- Aggregated Histograms with Error Bars ---
    # Aggregated Δt with error bars and exponential fit
    if aggregated['delta_t']:
        all_dt = np.concatenate(aggregated['delta_t'])
        dt_bins = np.linspace(dt_min, dt_max, bins+1)
        dt_hist, dt_edges = np.histogram(all_dt, bins=dt_bins)
        dt_centers = 0.5*(dt_edges[:-1] + dt_edges[1:])
        dt_err = np.sqrt(dt_hist)

        # Perform linear fit on log-counts: ln(N) = intercept + slope * t
        # define your desired fit window:
        t_low, t_high = 2500, 10000   # in ns

        # build mask: only positive bins, inside [t_low, t_high]
        fit_mask = (
            (dt_centers >= t_low) &
            (dt_centers <= t_high) &
            (dt_hist    > 0)
        )

        # pick out the points to fit
        fit_x = dt_centers[fit_mask]
        fit_y = np.log(dt_hist[fit_mask])

        # do the linear regression
        (slope, intercept), cov = np.polyfit(fit_x, fit_y, 1, cov=True)
        slope_err = np.sqrt(cov[0,0])
        tau = -1.0/slope
        tau_err = slope_err/(slope**2)
        fit_line = np.exp(intercept + slope*dt_centers)
        
        # Plot errorbar and fit
        plt.errorbar(dt_centers, dt_hist, yerr=dt_err, fmt='o', label='Data')
        plt.plot(dt_centers, fit_line, '--', label=f'Fit τ={tau:.1f}±{tau_err:.1f} ns')
        #shade the fit region, between t_low and t_high
        plt.axvspan(t_low, t_high, color='gray', alpha=0.2, label='Fit Range')
        plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'Aggregated Δt ({start_run}-{end_run})')
        if logscale: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        plt.savefig(output_dir/'aggregated_delta_t.png'); plt.close()
        save_pickle({'centers':dt_centers,'hist':dt_hist,'errors':dt_err,'fit_slope':slope,'fit_intercept':intercept,'tau':tau},
                    output_dir/'aggregated_delta_t.pkl')
        print(f"Saved aggregated Δt histogram with fit to {output_dir/'aggregated_delta_t.png'}")

    # Aggregated Total Charge
    if aggregated['sum_area_cut']:
        all_sa = np.concatenate(aggregated['sum_area_cut'])
        sa_bins = np.linspace(area_cut[0], area_cut[1], bins + 1)
        sa_hist, sa_edges = np.histogram(all_sa, bins=sa_bins)
        sa_centers = (sa_edges[:-1] + sa_edges[1:]) / 2
        sa_err = np.sqrt(sa_hist)
        plt.errorbar(sa_centers, sa_hist, yerr=sa_err, fmt='o', label=f'Runs {start_run}-{end_run}')
        plt.xlabel('Total Charge (ADC)'); plt.ylabel('Counts'); plt.title(f'Aggregated Total Charge ({start_run}-{end_run})')
        if logscale: plt.yscale('log')
        plt.legend(); plt.grid(which='both'); plt.tight_layout()
        plt.savefig(output_dir/'aggregated_sum_area.png'); plt.close()
        save_pickle({'hist': sa_hist, 'bin_centers': sa_centers, 'errorbars': sa_err}, output_dir/'aggregated_sum_area.pkl')
        print(f"Saved aggregated total charge histogram with error bars to {output_dir/'aggregated_sum_area.png'}")

if __name__ == '__main__':
    main()