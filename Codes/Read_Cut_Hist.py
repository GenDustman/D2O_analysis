#!/usr/bin/env python3
"""
Refactored script for processing ROOT files and saving histograms with detailed comments.
Generates general and cut histograms for triggerBits, sum_area, multiplicity,
and computes Δt and sum_area histograms after veto-based selections.
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
    Ensure that the given directory exists. Create it (and parents) if necessary.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(data: dict, path: Path):
    """
    Save a Python dictionary to a pickle file at the specified path.

    Parameters:
        data (dict): The data to serialize.
        path (Path): Path where pickle file will be written.
    """
    with path.open('wb') as f:
        pickle.dump(data, f)


def plot_histogram(
    arrays,
    labels,
    bins,
    img_path,
    title,
    xlabel,
    logscale=True,
    figsize=(10, 6)
):
    """
    Plot one or more histograms on the same axes and save the figure.

    Parameters:
        arrays (list of np.ndarray): Data arrays to histogram.
        labels (list of str): Legends for each histogram.
        bins (array-like): Bin edges.
        img_path (Path): Output image path.
        title (str): Plot title.
        xlabel (str): X-axis label.
        logscale (bool): If True, set y-axis to log scale.
        figsize (tuple): Figure size in inches.

    Returns:
        list of tuples: Each tuple is (counts, edges) returned by np.histogram.
    """
    plt.figure(figsize=figsize)
    outputs = []
    # Loop over each dataset and label
    for data, label in zip(arrays, labels):
        # Plot histogram and capture counts + bin edges
        counts, edges, _ = plt.hist(
            data,
            bins=bins,
            alpha=0.7,
            edgecolor='black',
            label=label
        )
        outputs.append((counts, edges))

    # Labeling and styling
    plt.xlabel(xlabel)
    plt.ylabel('Events')
    plt.title(title)

    # Optionally use log scale on y-axis
    if logscale:
        plt.yscale('log')

    plt.legend()
    plt.minorticks_on()
    # Grid for major/minor ticks
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()

    # Save and close
    plt.savefig(img_path)
    plt.close()
    return outputs


def save_trigger_bits_histogram(df, img_path, pkl_path, logscale=True):
    """
    Generate and save a histogram of the 'triggerBits' column.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'triggerBits'.
        img_path (Path): Where to save the PNG figure.
        pkl_path (Path): Where to save the histogram data as pickle.
        logscale (bool): If True, use log scale on y-axis.
    """
    # Extract numpy array
    data = df['triggerBits'].to_numpy()
    # Define integer bins from 0 to 35
    bins = np.arange(0, 36)
    # Plot and retrieve counts + edges
    counts, edges = plot_histogram(
        [data],
        ['triggerBits'],
        bins,
        img_path,
        'Histogram of Trigger Bits',
        'Trigger Bits',
        logscale=logscale
    )[0]

    # Save histogram data
    save_pickle({'hist': counts, 'bins': edges}, pkl_path)
    print(f"Saved trigger bits histogram to {img_path} and data to {pkl_path}")


def save_comparison_histogram(
    df,
    column,
    trigger_val,
    img_path,
    pkl_path,
    bins,
    logscale=True
):
    """
    Generate and save comparison histograms for all entries vs. those with a specific triggerBits value.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to histogram (e.g. 'sum_area', 'multiplicity').
        trigger_val (int): Value of triggerBits to subset for comparison.
        img_path (Path): Output image path.
        pkl_path (Path): Output pickle path for histogram data.
        bins (array-like): Bin edges.
        logscale (bool): Use log scale on y-axis.
    """
    all_data  = df[column].to_numpy()
    trig_data = df.loc[df['triggerBits'] == trigger_val, column].to_numpy()

    outputs = plot_histogram(
        [all_data, trig_data],
        ['All', f'triggerBits == {trigger_val}'],
        bins,
        img_path,
        f'Histogram of {column}',
        column,
        logscale=logscale,
        figsize=(12, 8)
    )

    # Save raw histogram data
    save_pickle(
        {
            'all_data': outputs[0][0],
            'trigger_data': outputs[1][0],
            'bins': outputs[0][1]
        },
        pkl_path
    )
    print(f"Saved comparison histogram for {column} to {img_path} and data to {pkl_path}")


def cut_and_save_histograms(
    df,
    delta_t_cut,
    area_cut,
    bins_num,
    save_dir,
    run_number,
    multiplicity_adc,
    multiplicity_cut,
    logscale=True
):
    """
    Apply sequential cuts on Δt and total charge, then save errorbar plots and raw data.
    Always produce output files (even if counts are zero).

    Parameters:
        df (pd.DataFrame): Data including 'nsTime', 'triggerBits', 'sum_area', 'multiplicity'.
        delta_t_cut (tuple): (min, max) ns range for Δt.
        area_cut (tuple): (min, max) ADC range for sum_area.
        bins_num (int): Number of bins for histograms.
        save_dir (Path): Directory where outputs are written.
        run_number (int): Identifier for labeling.
        multiplicity_adc (int): ADC threshold for counting channels.
        multiplicity_cut (int): Minimum number of channels to qualify as veto event.
        logscale (bool): Use log scale on y-axis.

    Returns:
        (delta_t_values, sum_area_values)
    """
    ensure_dir(save_dir)

    # Identify muon times for reference
    muon_times = df.loc[df['triggerBits'] >= 32, 'nsTime'].values
    # Select events immediately after veto (triggerBits==2, multiplicity>cut)
    after_veto = df[(df['triggerBits'] == 2) & (df['multiplicity'] > multiplicity_cut)].copy()
    print(f"Run {run_number}: after-veto events: {len(after_veto)}")

    # Compute Δt relative to most recent muon
    times = after_veto['nsTime'].values
    idx = np.searchsorted(muon_times, times, side='right')
    delta_t = np.full_like(times, np.nan, dtype=float)
    valid   = idx > 0
    delta_t[valid] = times[valid] - muon_times[idx[valid] - 1]
    after_veto['delta_t'] = delta_t

    # Apply Δt cut
    dt_min, dt_max = delta_t_cut
    sel_dt = after_veto[(after_veto['delta_t'] >= dt_min) & (after_veto['delta_t'] <= dt_max)]
    print(f"Run {run_number}: after Δt cut: {len(sel_dt)}")

    # Apply sum_area cut
    s_min, s_max = area_cut
    sel = sel_dt[(sel_dt['sum_area'] >= s_min) & (sel_dt['sum_area'] <= s_max)]
    print(f"Run {run_number}: after sum_area cut: {len(sel)}")

    # Define bins for histograms
    dt_bins = np.linspace(dt_min, dt_max, bins_num + 1)
    s_bins  = np.linspace(s_min,  s_max,  bins_num + 1)

    # Δt histogram (errorbar style)
    dt_hist, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers        = (dt_edges[:-1] + dt_edges[1:]) / 2
    dt_err            = np.sqrt(dt_hist)
    save_pickle({'hist': dt_hist, 'bin_centers': dt_centers, 'errorbars': dt_err}, save_dir/'delta_t_histogram.pkl')
    plt.errorbar(dt_centers, dt_hist, yerr=dt_err, fmt='o',
                 label=f"Run {run_number}: Δt {dt_min}-{dt_max}, area {s_min}-{s_max}")
    plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title('Δt Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.minorticks_on(); plt.grid(which='both'); plt.tight_layout()
    plt.savefig(save_dir/'delta_t_histogram.png'); plt.close()

    # sum_area histogram (errorbar style)
    s_hist, s_edges = np.histogram(sel['sum_area'], bins=s_bins)
    s_centers       = (s_edges[:-1] + s_edges[1:]) / 2
    s_err           = np.sqrt(s_hist)
    save_pickle({'hist': s_hist, 'bin_centers': s_centers, 'errorbars': s_err}, save_dir/'sum_area_histogram.pkl')
    plt.errorbar(s_centers, s_hist, yerr=s_err, fmt='o',
                 label=f"Run {run_number}: Δt {dt_min}-{dt_max}, area {s_min}-{s_max}")
    plt.xlabel('Total Charge (ADC)'); plt.ylabel('Counts'); plt.title('Total Charge Histogram')
    if logscale: plt.yscale('log')
    plt.legend(); plt.minorticks_on(); plt.grid(which='both'); plt.tight_layout()
    plt.savefig(save_dir/'sum_area_histogram.png'); plt.close()

    # Return raw arrays for aggregation
    return sel['delta_t'].values, sel['sum_area'].values


def process_run(
    run,
    data_dir,
    output_dir,
    delta_t_cut,
    area_cut,
    bins_cut,
    multiplicity_adc,
    multiplicity_cut
):
    """
    Process a single run: read ROOT tree, build DataFrame, save general and cut histograms.

    Parameters:
        run (int): Run number to process.
        data_dir (Path): Base directory for ROOT files.
        output_dir (Path): Where to write outputs.
        delta_t_cut, area_cut: Tuples defining cut ranges.
        bins_cut (int): Number of bins for cut histograms.
        multiplicity_adc (int): ADC threshold for multiplicity.
        multiplicity_cut (int): Minimum channel count for veto events.

    Returns:
        tuple of arrays or None: Δt and sum_area arrays for aggregated plotting.
    """
    print(f"Processing run {run}")
    root_path = data_dir/f"run{run}_processed_v5.root"
    if not root_path.exists():
        print(f"File not found: {root_path}")
        return None

    # Read in chunks to manage memory
    dfs = []
    for chunk in uproot.open(root_path)['tree'].iterate(
        ['eventID', 'nsTime', 'triggerBits', 'area'],
        library='ak', step_size='100 MB'
    ):
        arr = ak.to_numpy(chunk['area'])
        # Build DataFrame for this chunk
        df_chunk = pd.DataFrame({
            'eventID'    : ak.to_numpy(chunk['eventID']),
            'nsTime'     : ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            'sum_area'   : np.sum(arr[:, :12], axis=1),
            'multiplicity': np.sum(arr[:, :12] > multiplicity_adc, axis=1)
        })
        dfs.append(df_chunk)

    # Concatenate all chunks
    if not dfs:
        print(f"No data for run {run}")
        return None
    df = pd.concat(dfs, ignore_index=True)
    # Save raw DataFrame
    df.to_pickle(output_dir/f"run{run}_data.pkl")

    # Prepare subdirectories
    hist_dir = output_dir/f"run{run}"/'histograms'
    cut_dir  = output_dir/f"run{run}"/'cuthist'
    ensure_dir(hist_dir)
    ensure_dir(cut_dir)

    # General histograms
    save_trigger_bits_histogram(df, hist_dir/f"run{run}_triggerBits.png", hist_dir/f"run{run}_triggerBits.pkl")
    save_comparison_histogram(df, 'sum_area', 2, hist_dir/f"run{run}_sum_area.png", hist_dir/f"run{run}_sum_area.pkl", np.linspace(0, 100000, bins_cut+1))
    save_comparison_histogram(df, 'multiplicity', 2, hist_dir/f"run{run}_multiplicity.png", hist_dir/f"run{run}_multiplicity.pkl", np.arange(0, 13))

    # Cut histograms
    return cut_and_save_histograms(
        df, delta_t_cut, area_cut, bins_cut, cut_dir,
        run, multiplicity_adc, multiplicity_cut
    )


def main():
    """
    Entry point: parse arguments and loop over runs.
    """
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_run> <end_run>")
        sys.exit(1)

    start_run, end_run = map(int, sys.argv[1:])
    data_dir  = Path('/raid1/genli/Data_D2O')
    output_dir = data_dir/f"runs_{start_run}_{end_run}"
    ensure_dir(output_dir)

    # Configuration parameters
    delta_t_cut      = (0, 20000)    # Δt range in ns
    area_cut         = (0, 100000)   # sum_area ADC range
    bins_cut         = 20            # Number of bins for cut histograms
    multiplicity_adc = 3000          # ADC threshold for channel count
    multiplicity_cut = 0             # Minimum channel count for veto events

    # Aggregated results storage
    aggregated = {'delta_t': [], 'sum_area_cut': []}

    # Process each run in range
    for run in range(start_run, end_run + 1):
        result = process_run(
            run, data_dir, output_dir,
            delta_t_cut, area_cut, bins_cut,
            multiplicity_adc, multiplicity_cut
        )
        if result is not None:
            dt_vals, sa_vals = result
            aggregated['delta_t'].append(dt_vals)
            aggregated['sum_area_cut'].append(sa_vals)

    # (Optional) Further aggregation and plotting can be implemented here

if __name__ == '__main__':
    main()
