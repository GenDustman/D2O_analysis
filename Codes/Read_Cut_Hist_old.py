#!/usr/bin/env python3
"""
Refactored script for processing ROOT files and saving histograms with modular functions.
"""
import sys
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import awkward as ak

def ensure_dir(path: Path):
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(data: dict, path: Path):
    """Save a dictionary as a pickle file."""
    with path.open('wb') as f:
        pickle.dump(data, f)


def plot_histogram(
    arrays: list[np.ndarray],
    labels: list[str],
    bins: np.ndarray,
    img_path: Path,
    title: str,
    xlabel: str,
    ylabel: str = 'Events',
    logscale: bool = True,
    **kwargs
):
    """
    Plot one or more histograms on the same axes and save to file.
    Returns the counts and bin edges for each array.
    """
    plt.figure(figsize=kwargs.get('figsize', (10, 6)))
    hist_outputs = []
    for data, label in zip(arrays, labels):
        counts, bin_edges, _ = plt.hist(
            data,
            bins=bins,
            alpha=0.7,
            edgecolor='black',
            label=label
        )
        hist_outputs.append((counts, bin_edges))

    plt.xlabel(xlabel, fontsize=kwargs.get('xlabel_size', 14))
    plt.ylabel(ylabel, fontsize=kwargs.get('ylabel_size', 14))
    plt.title(title, fontsize=kwargs.get('title_size', 16))
    if logscale:
        plt.yscale('log')
    if labels:
        plt.legend(fontsize=kwargs.get('legend_size', 12))
    plt.minorticks_on()
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    return hist_outputs


def save_trigger_bits_histogram(df: pd.DataFrame, img_path: Path, pkl_path: Path, logscale: bool = True):
    """Generate a histogram of the 'triggerBits' column."""
    data = df['triggerBits'].to_numpy()
    bins = np.arange(0, 36)
    counts, bin_edges = plot_histogram(
        [data],
        ['triggerBits'],
        bins,
        img_path,
        title='Histogram of Trigger Bits',
        xlabel='Trigger Bits',
        logscale=logscale
    )[0]

    save_pickle({'hist': counts, 'bins': bin_edges}, pkl_path)
    print(f"Saved trigger bits histogram to {img_path} and data to {pkl_path}")


def save_comparison_histogram(
    df: pd.DataFrame,
    column: str,
    mask: pd.Series,
    img_path: Path,
    pkl_path: Path,
    bins: np.ndarray,
    logscale: bool = True
):
    """
    Compare histograms of a full dataset vs. a masked subset.
    """
    full = df[column].to_numpy()
    subset = df.loc[mask, column].to_numpy()

    outputs = plot_histogram(
        [full, subset],
        ['All', f"{column} with mask"],
        bins,
        img_path,
        title=f"Histogram of {column}",
        xlabel=column,
        logscale=logscale,
        figsize=(12, 8)
    )

    save_pickle({
        'all_data': outputs[0][0],
        'trigger_data': outputs[1][0],
        'bins': outputs[0][1]
    }, pkl_path)
    print(f"Saved comparison histogram for {column} to {img_path} and data to {pkl_path}")


def compute_delta_t(
    df: pd.DataFrame,
    muon_bits: int,
    veto_bits: int,
    multiplicity_cut: int
) -> pd.DataFrame:
    """
    Compute time difference between veto events and last muon event.
    Returns a DataFrame of events that pass the veto mask, with a new 'delta_t' column.
    """
    muon_mask = df['triggerBits'] >= muon_bits
    veto_mask = (df['triggerBits'] == veto_bits) & (df['multiplicity'] > multiplicity_cut)
    muon_times = df.loc[muon_mask, 'nsTime'].values
    events = df.loc[veto_mask, :].copy()

    times = events['nsTime'].values
    idx = np.searchsorted(muon_times, times, side='right')
    delta_t = np.full(times.shape, np.nan)
    valid = idx > 0
    delta_t[valid] = times[valid] - muon_times[idx[valid] - 1]
    events['delta_t'] = delta_t
    return events


def save_cut_histograms(
    events: pd.DataFrame,
    delta_t_range: tuple[float, float],
    area_range: tuple[float, float],
    bins: int,
    save_dir: Path,
    run_label: str,
    logscale: bool = True
):
    """
    Apply cuts on delta_t and sum_area, then save errorbar histograms and data.
    """
    dt_min, dt_max = delta_t_range
    s_min, s_max = area_range
    # apply cuts
    sel = events.dropna(subset=['delta_t'])
    sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
    sel = sel[(sel['sum_area'] >= s_min) & (sel['sum_area'] <= s_max)]

    if sel.empty:
        print(f"{run_label}: no events after cuts.")
        return None, None

    # Delta T histogram
    dt_bins = np.linspace(dt_min, dt_max, bins + 1)
    dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
    dt_centers = (dt_edges[:-1] + dt_edges[1:]) / 2
    dt_err = np.sqrt(dt_counts)
    save_pickle(
        {'hist': dt_counts, 'bin_centers': dt_centers, 'errorbars': dt_err},
        save_dir / 'delta_t_histogram.pkl'
    )
    plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o',
                 label=f"{run_label} Δt {dt_min}-{dt_max}, area {s_min}-{s_max}")
    plt.xlabel('Δt (ns)')
    plt.ylabel('Counts')
    plt.title('Δt Histogram')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'delta_t_histogram.png')
    plt.close()

    # Sum area histogram
    s_bins = np.linspace(s_min, s_max, bins + 1)
    s_counts, s_edges = np.histogram(sel['sum_area'], bins=s_bins)
    s_centers = (s_edges[:-1] + s_edges[1:]) / 2
    s_err = np.sqrt(s_counts)
    save_pickle(
        {'hist': s_counts, 'bin_centers': s_centers, 'errorbars': s_err},
        save_dir / 'sum_area_histogram.pkl'
    )
    plt.errorbar(s_centers, s_counts, yerr=s_err, fmt='o',
                 label=f"{run_label} Δt {dt_min}-{dt_max}, area {s_min}-{s_max}")
    plt.xlabel('Total Charge (ADC)')
    plt.ylabel('Counts')
    plt.title('Total Charge Histogram')
    if logscale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'sum_area_histogram.png')
    plt.close()

    return sel['delta_t'].values, sel['sum_area'].values


def process_run(
    run: int,
    data_dir: Path,
    output_dir: Path,
    delta_t_cut: tuple[float, float],
    area_cut: tuple[float, float],
    bins_cut: int,
    adc_threshold: int,
    multiplicity_cut: int
):
    """Process a single run: read ROOT, save individual and cut histograms, return raw arrays."""
    print(f"Processing run {run}")
    file_path = data_dir / f"run{run}_processed_v5.root"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None

    # Read and build DataFrame
    dfs = []
    for chunk in uproot.open(file_path)['tree'].iterate(
        ['eventID', 'nsTime', 'triggerBits', 'area'], library='ak', step_size='100 MB'
    ):
        arr = ak.to_numpy(chunk['area'])
        df_chunk = pd.DataFrame({
            'eventID': ak.to_numpy(chunk['eventID']),
            'nsTime': ak.to_numpy(chunk['nsTime']),
            'triggerBits': ak.to_numpy(chunk['triggerBits']),
            'sum_area': np.sum(arr[:, :12], axis=1),
            'multiplicity': np.sum(arr[:, :12] > adc_threshold, axis=1)
        })
        dfs.append(df_chunk)

    if not dfs:
        print(f"No data for run {run}")
        return None
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(output_dir / f"run{run}_data.pkl")

    # Create subfolders
    hist_dir = output_dir / f"run{run}" / "histograms"
    cut_dir = output_dir / f"run{run}" / "cuts"
    ensure_dir(hist_dir)
    ensure_dir(cut_dir)

    # Save simple hist
    save_trigger_bits_histogram(
        df,
        hist_dir / f"run{run}_triggerBits.png",
        hist_dir / f"run{run}_triggerBits.pkl"
    )

    # Save comparison histograms
    bins_area = np.linspace(0, 100000, bins_cut + 1)
    save_comparison_histogram(
        df,
        'sum_area',
        df['triggerBits'] == 2,
        hist_dir / f"run{run}_sum_area.png",
        hist_dir / f"run{run}_sum_area.pkl",
        bins_area
    )
    bins_mult = np.arange(0, 13)
    save_comparison_histogram(
        df,
        'multiplicity',
        df['triggerBits'] == 2,
        hist_dir / f"run{run}_multiplicity.png",
        hist_dir / f"run{run}_multiplicity.pkl",
        bins_mult
    )

    # Compute and save cut histograms
    events = compute_delta_t(
        df,
        muon_bits=32,
        veto_bits=2,
        multiplicity_cut=multiplicity_cut
    )
    return save_cut_histograms(
        events,
        delta_t_cut,
        area_cut,
        bins_cut,
        cut_dir,
        f"Run {run}"
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_run> <end_run>")
        sys.exit(1)

    start_run, end_run = map(int, sys.argv[1:])
    data_dir = Path('/raid1/genli/Data_D2O')
    output_dir = data_dir / f"runs_{start_run}_{end_run}"
    ensure_dir(output_dir)

    # Configuration
    delta_t_cut      = (0, 20000)   # ns
    area_cut         = (0, 100000)  # ADC units
    bins_cut         = 20           # histogram bins
    adc_threshold    = 30_00         # ADC threshold for channel firing, 1 p.e. = 100 ADC
    multiplicity_cut = 0            # min number of fired channels

    aggregated = {
        'delta_t': [],
        'sum_area_cut': []
    }

    for run in range(start_run, end_run + 1):
        result = process_run(
            run,
            data_dir,
            output_dir,
            delta_t_cut,
            area_cut,
            bins_cut,
            adc_threshold,
            multiplicity_cut
        )
        if result:
            dt_vals, sa_vals = result
            aggregated['delta_t'].append(dt_vals)
            aggregated['sum_area_cut'].append(sa_vals)

    # Further aggregated plotting can be added here similarly.

if __name__ == '__main__':
    main()
