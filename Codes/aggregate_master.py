#!/usr/bin/env python3
"""
Master Aggregation Script for D2O Analysis

Reads all sub-job outputs and creates final, grand-aggregated results.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# --- Import shared configuration parameters ---
import config

# --- Import plotting and analysis functions from the new processing script ---
from Read_Cut_Hist_D2O_multi import (
    ensure_dir,
    aggregate_plots,
    plot_correlation_maps,
    fit_and_plot_low_light,
    plot_sipm_histograms,
    plot_veto_efficiency,
    plot_histogram
)

def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate_master.py <top_level_analysis_directory>")
        sys.exit(1)

    top_dir = Path(sys.argv[1])
    if not top_dir.is_dir():
        print(f"Error: Directory not found at {top_dir}")
        sys.exit(1)

    # Output directory for the final, master results
    master_output_dir = top_dir / "MASTER_RESULTS"
    ensure_dir(master_output_dir)
    print(f"Master output will be saved to: {master_output_dir}")

    subjob_dirs = sorted(list(top_dir.glob("subjob_*")))
    if not subjob_dirs:
        print("Error: No 'subjob_*' directories found. Did the jobs run correctly?")
        sys.exit(1)
        
    print(f"Found {len(subjob_dirs)} sub-job directories to aggregate.")

    # --- Initialize data containers ---
    all_dt, all_pe, all_mult = [], [], []
    all_ll_areas = []
    all_sipm_events_df, all_pe_trig2_df, all_pe_trig2_or_34_df = [], [], []

    # --- Loop over sub-jobs and collect data ---
    for sub_dir in subjob_dirs:
        print(f"Processing {sub_dir.name}...")
        if (sub_dir / 'aggregated_delta_t.npy').exists():
            all_dt.append(np.load(sub_dir / 'aggregated_delta_t.npy'))
            all_pe.append(np.load(sub_dir / 'aggregated_total_pe.npy'))
            all_mult.append(np.load(sub_dir / 'aggregated_multiplicity.npy'))
        if (sub_dir / 'aggregated_low_light_areas.npy').exists():
            all_ll_areas.append(np.load(sub_dir / 'aggregated_low_light_areas.npy'))
        if (sub_dir / 'aggregated_sipm_events.pkl').exists():
            all_sipm_events_df.append(pd.read_pickle(sub_dir / 'aggregated_sipm_events.pkl'))
        if (sub_dir / 'aggregated_pe_trig2.pkl').exists():
            all_pe_trig2_df.append(pd.read_pickle(sub_dir / 'aggregated_pe_trig2.pkl'))
        if (sub_dir / 'aggregated_pe_trig2_or_34.pkl').exists():
            all_pe_trig2_or_34_df.append(pd.read_pickle(sub_dir / 'aggregated_pe_trig2_or_34.pkl'))

    # Extract M1/M2 and run range from directory name for plot labels
    dir_name_parts = top_dir.name.split('_')
    run_range_str = dir_name_parts[1]
    m1_or_m2 = dir_name_parts[2]
    agg_label = f"Master Runs {run_range_str}"

    # --- Perform Grand Aggregation and Plotting (using imported config) ---

    # 1. Delta_t, Total_PE, and Tau Fit
    if all_dt:
        print("Aggregating delta_t, total_pe, and fitting for tau...")
        master_aggregated_data = {
            'delta_t': [np.concatenate(all_dt)],
            'total_pe': [np.concatenate(all_pe)],
            'multiplicity': [np.concatenate(all_mult)]
        }
        aggregate_plots(
            master_aggregated_data, config.DELTA_T_CUT, config.PE_CUT, config.BINS, 
            config.TAU_FIT_WINDOW, master_output_dir, m1_or_m2, agg_label, 
            config.LOGSCALE_DT_AGG, config.LOGSCALE_PE_AGG, config.DO_TAU_FIT
        )
        master_corr_df = pd.DataFrame({
            'delta_t': np.concatenate(all_dt), 'total_pe': np.concatenate(all_pe),
            'multiplicity': np.concatenate(all_mult)
        })
        plot_correlation_maps(master_corr_df, master_output_dir, agg_label, m1_or_m2)

    # 2. Veto Efficiency
    if all_pe_trig2_df and all_pe_trig2_or_34_df:
        print("Aggregating veto efficiency data...")
        master_pe_trig2 = pd.concat(all_pe_trig2_df, ignore_index=True)
        master_pe_trig2_or_34 = pd.concat(all_pe_trig2_or_34_df, ignore_index=True)
        
        filename_label = agg_label.replace(" ", "_").replace("-", "_")
        
        plot_histogram(
            [master_pe_trig2_or_34, master_pe_trig2], ['Trig=2 or 34', 'Trig=2'],
            np.linspace(*config.PE_CUT, config.BINS + 1),
            master_output_dir / f"{filename_label}_{m1_or_m2}_total_pe_comparison_master.png",
            f'Master Total PE Comparison {agg_label}', 'Total P.E.', m1_or_m2, logscale=True
        )

        veto_img_path = master_output_dir / f"{filename_label}_{m1_or_m2}_veto_efficiency_master.png"
        veto_pkl_path = master_output_dir / f"{filename_label}_{m1_or_m2}_veto_efficiency_master.pkl"
        plot_veto_efficiency(
            master_pe_trig2.to_numpy(), master_pe_trig2_or_34.to_numpy(),
            config.VETO_BINS, config.VETO_RANGE, config.PE_CUT, veto_img_path, 
            veto_pkl_path, f"Master Veto Efficiency {agg_label}", m1_or_m2
        )

    # 3. Low-Light Fits
    if all_ll_areas:
        master_ll_areas = np.concatenate(all_ll_areas, axis=0)
        fit_and_plot_low_light(master_ll_areas, master_output_dir, agg_label, 
                               m1_or_m2, hist_range=config.LOW_LIGHT_FIT_RANGE)

    # 4. SiPM Histograms
    if all_sipm_events_df:
        master_sipm_df = pd.concat(all_sipm_events_df, ignore_index=True)
        plot_sipm_histograms(master_sipm_df, master_output_dir, agg_label, 
                             m1_or_m2, **config.SIPM_HIST_CONFIG)

    print("\n--- Master Aggregation Complete ---")

if __name__ == '__main__':
    main()