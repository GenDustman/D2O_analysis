#!/usr/bin/env python
# %%
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uproot
import awkward as ak
import pandas as pd
from scipy.optimize import curve_fit
import sys
import pickle
import os

# %%
def save_trigger_bits_histogram(df, img_path, pkl_path, logscale=True):
    """
    Generates and saves a histogram of triggerBits.
    The y-axis will be in log scale if logscale=True.
    Saves the histogram figure as a PNG and the histogram data (counts and bins) as a pickle file.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        img_path (str): File path to save the histogram image.
        pkl_path (str): File path to save the histogram data as a pickle file.
        logscale (bool): Whether to use log scale on the y-axis.
    """
    trigger_bits = df['triggerBits'].to_numpy()
    bins_array = np.linspace(0, 35, 36)
    
    plt.figure(figsize=(8, 5))
    hist, bins, _ = plt.hist(trigger_bits, bins=bins_array, edgecolor='black', alpha=0.7)
    plt.xlabel("Trigger Bits")
    plt.ylabel("Events")
    if logscale:
        plt.yscale("log")
    plt.title("Histogram of Trigger Bits")
    plt.minorticks_on()
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(img_path)
    plt.show()
    
    hist_data = {"hist": hist, "bins": bins}
    with open(pkl_path, "wb") as f:
        pickle.dump(hist_data, f)
    
    print(f"Trigger Bit Histogram image saved to: {img_path}")
    print(f"Trigger Bit Histogram data saved to: {pkl_path}")

# %%
def save_all_histogram(df, column_name, trigger_column, trigger_value, img_path, pkl_path, bins_num=100, logscale=True):
    """
    Generates and saves histograms of the specified column, comparing all data vs. a subset 
    where the trigger condition is met. Saves the histogram figure as a JPEG and the histogram 
    data (raw arrays and binning) as a pickle file.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        column_name (str): The column name to create the histograms for.
        trigger_column (str): The column used to filter data based on a trigger value.
        trigger_value (int): The trigger value to filter by.
        img_path (str): File path to save the histogram image.
        pkl_path (str): File path to save the histogram data as a pickle file.
        bins_num (int): Number of bins to use.
        logscale (bool): Whether to use log scale on the y-axis.
    """
    all_data = df[column_name].to_numpy()
    trigger_data = df.loc[df[trigger_column] == trigger_value, column_name].to_numpy()
    
    plt.figure(figsize=(12, 8))
    bins_arr = np.linspace(0, 100000, bins_num + 1)
    plt.hist(all_data, bins=bins_arr, alpha=0.7, edgecolor='black', label='All triggerBits')
    plt.hist(trigger_data, bins=bins_arr, alpha=0.7, edgecolor='black', label=f'{trigger_column} = {trigger_value}')
    
    plt.xlabel("Total Charge (ADC)", fontsize=25)
    plt.ylabel("Events", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Histogram of Total Charge (ADC)", fontsize=25)
    if logscale:
        plt.yscale("log")
    plt.legend(fontsize=20)
    plt.minorticks_on()
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
    plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
    plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
    plt.xlim(0, 100000)
    plt.tight_layout()
    
    plt.savefig(img_path)
    plt.show()
    
    hist_data = {"all_data": all_data, "trigger_data": trigger_data, "bins": bins_arr}
    with open(pkl_path, "wb") as f:
        pickle.dump(hist_data, f)
    
    print(f"Energy Histogram image saved to: {img_path}")
    print(f"Energy Histogram data saved to: {pkl_path}")

# %%
def Cut_DeltaT_Totalcharge(df, deltaT_cut, sum_area_cut, bins_num, save_folder, run_number, logscale=True):
    """
    For after-veto events (triggerBits==2), compute Δt from the previous muon event (triggerBits>=32), 
    apply a Δt cut and a total charge (sum_area) cut, and then produce two histograms (Δt and sum_area).
    The y-axis is set to log scale if logscale=True.
    The legend indicates the run number and the applied cut values.
    
    The function saves the plots and the histogram data (counts, bin centers, error bars) as pickle files.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'triggerBits', 'nsTime', and 'sum_area'.
        deltaT_cut (tuple): (dt_min, dt_max) in ns.
        sum_area_cut (tuple): (s_min, s_max) in ADC units.
        bins_num (int): Number of bins to use.
        save_folder (str): Folder to save the output.
        run_number (int): Run number (for labeling the plots).
        logscale (bool): Whether to use log scale on the y-axis.
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # Identify muon and after-veto events and compute Δt.
    muon_mask = df['triggerBits'] >= 32
    after_veto_mask = df['triggerBits'] == 2
    muon_times = df.loc[muon_mask, 'nsTime'].values
    after_veto_df = df.loc[after_veto_mask].copy()
    event_times = after_veto_df['nsTime'].values
    insertion_indices = np.searchsorted(muon_times, event_times, side='right')
    delta_t = np.full_like(event_times, np.nan, dtype=float)
    valid = insertion_indices > 0
    delta_t[valid] = event_times[valid] - muon_times[insertion_indices[valid] - 1]
    after_veto_df['delta_t'] = delta_t
    
    dt_min, dt_max = deltaT_cut
    s_min, s_max = sum_area_cut
    selected = after_veto_df[
        (after_veto_df['delta_t'] >= dt_min) &
        (after_veto_df['delta_t'] <= dt_max) &
        (after_veto_df['sum_area'] >= s_min) &
        (after_veto_df['sum_area'] <= s_max)
    ]
    if selected.empty:
        print(f"Run {run_number}: No events passed the selection cuts.")
        return None, None  # Return None if no selected events.
    
    # Δt histogram.
    dt_bins = np.linspace(dt_min, dt_max, bins_num + 1)
    dt_hist, dt_bin_edges = np.histogram(selected['delta_t'], bins=dt_bins)
    dt_centers = (dt_bin_edges[:-1] + dt_bin_edges[1:]) / 2
    dt_err = np.sqrt(dt_hist)
    
    dt_data = {"hist": dt_hist, "bin_centers": dt_centers, "errorbars": dt_err}
    dt_pkl_path = os.path.join(save_folder, 'delta_t_histogram.pkl')
    with open(dt_pkl_path, 'wb') as f:
        pickle.dump(dt_data, f)
    
    plt.figure(figsize=(10, 7))
    plt.errorbar(dt_centers, dt_hist, yerr=dt_err, fmt='o', label=f"Run {run_number}: Δt cut {dt_min}-{dt_max} ns")
    plt.xlabel("Δt (ns)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.title("Δt Histogram (Selected Events)", fontsize=16)
    if logscale:
        plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    dt_plot_path = os.path.join(save_folder, 'delta_t_histogram.png')
    plt.savefig(dt_plot_path)
    plt.close()
    
    # sum_area histogram.
    s_bins = np.linspace(s_min, s_max, bins_num + 1)
    s_hist, s_bin_edges = np.histogram(selected['sum_area'], bins=s_bins)
    s_centers = (s_bin_edges[:-1] + s_bin_edges[1:]) / 2
    s_err = np.sqrt(s_hist)
    
    s_data = {"hist": s_hist, "bin_centers": s_centers, "errorbars": s_err}
    s_pkl_path = os.path.join(save_folder, 'sum_area_histogram.pkl')
    with open(s_pkl_path, 'wb') as f:
        pickle.dump(s_data, f)
    
    plt.figure(figsize=(10, 7))
    plt.errorbar(s_centers, s_hist, yerr=s_err, fmt='o', label=f"Run {run_number}: sum_area cut {s_min}-{s_max} ADC")
    plt.xlabel("Total Charge (ADC)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.title("Total Charge Histogram (Selected Events)", fontsize=16)
    if logscale:
        plt.yscale("log")
    plt.legend(fontsize=12)
    plt.tight_layout()
    s_plot_path = os.path.join(save_folder, 'sum_area_histogram.png')
    plt.savefig(s_plot_path)
    plt.close()
    
    print(f"Run {run_number}: Δt and Total Charge histograms (with cuts) saved in: {save_folder}")
    
    # Return the selected raw arrays for aggregated plotting.
    return selected['delta_t'].values, selected['sum_area'].values

# %%
def main():
    """
    Expects two command-line arguments: start_run and end_run.
    For each run between these two numbers (inclusive), the script processes the ROOT file, 
    generates and saves individual histograms, and collects raw data for aggregated histograms.
    Finally, it creates a "Total_run_<start_run>_<end_run>" folder containing aggregated plots 
    and histogram data.
    
    A logscale switch (default True) is used throughout.
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <start_run> <end_run>")
        sys.exit(1)
    start_run = int(sys.argv[1])
    end_run = int(sys.argv[2])
    
    # Set common cut values.
    deltaT_cut = (0, 20000)    # in ns
    sum_area_cut = (0, 100000)   # ADC units
    bins_num_cut = 100             # for cut histograms
    # Log scale switch for all functions.
    logscale = True
    
    # Folders for individual results.
    newcut_folder = '/raid1/genli/Data_D2O/'+f"run{start_run}_{end_run}_dt{deltaT_cut[0]}-{deltaT_cut[1]}_sa{sum_area_cut[0]}-{sum_area_cut[1]}"+'/'
    os.makedirs(newcut_folder, exist_ok=True) 
    # Lists to collect aggregated raw data.
    aggregated_trigger_bits = []
    aggregated_energy_all = []
    aggregated_energy_trigger = []
    aggregated_cut_deltaT = []
    aggregated_cut_sum_area = []
    
    # Process each run.
    for run in range(start_run, end_run+1):
        print(f"\nProcessing run: {run}")
        file_path = f"/raid1/genli/Data_D2O/run{run}_processed_v5.root"
        branches_to_read = ["eventID", "nsTime", "triggerBits", "pulseH", "area"]
        data_list = []
        
        try:
            with uproot.open(file_path) as file:
                tree = file["tree"]
                for chunk in tree.iterate(branches_to_read, library="ak", step_size="100 MB"):
                    scalar_data = {key: ak.to_numpy(chunk[key]) for key in ["eventID", "nsTime", "triggerBits"]}
                    array_data = {key: list(ak.to_numpy(chunk[key])) for key in ["pulseH", "area"]}
                    area_arrays = ak.to_numpy(chunk["area"])
                    scalar_data['sum_area'] = np.sum(area_arrays[:, 0:11], axis=1)
                    combined_data = {**scalar_data, **array_data}
                    data_list.append(pd.DataFrame(combined_data))
        except Exception as e:
            print(f"Error processing run {run}: {e}")
            continue
        
        if not data_list:
            print(f"Run {run}: No data found.")
            continue
        df_new = pd.concat(data_list, ignore_index=True)
        print(f"Run {run}: Finished processing ROOT file into DataFrame.")
        
        # Save DataFrame.
        output_path = f"/raid1/genli/Data_D2O/run{run}_data_new.pkl"
        df_new.to_pickle(output_path)
        print(f"Run {run}: DataFrame saved to: {output_path}")
        # Folder for histograms for this run.
        general_hist_folder = newcut_folder + f"run{run}_gneral/"
        os.makedirs(general_hist_folder, exist_ok=True)
        # Save trigger bits histogram.
        trig_img = os.path.join(general_hist_folder, f"run{run}_trigger_bits_histogram.png")
        trig_pkl = os.path.join(general_hist_folder, f"run{run}_trigger_bits_histogram.pkl")
        save_trigger_bits_histogram(df_new, trig_img, trig_pkl, logscale=logscale)
        
        # Save energy (total charge) histogram comparing all vs. trigger==2.
        energy_img = os.path.join(general_hist_folder, f"run{run}_sum_area_histogram.jpg")
        energy_pkl = os.path.join(general_hist_folder, f"run{run}_sum_area_histogram.pkl")
        save_all_histogram(df_new, "sum_area", "triggerBits", 2, energy_img, energy_pkl, bins_num=bins_num_cut, logscale=logscale)
        
        # Folder for cut histograms for this run.
        run_hist_folder = newcut_folder + f"run{run}_cuthist/"
        os.makedirs(run_hist_folder, exist_ok=True)
        cut_deltaT, cut_totalcharge = Cut_DeltaT_Totalcharge(df_new, deltaT_cut, sum_area_cut, bins_num_cut, run_hist_folder, run, logscale=logscale)
        
        # Collect raw data for aggregated histograms.
        aggregated_trigger_bits.append(df_new['triggerBits'].to_numpy())
        aggregated_energy_all.append(df_new['sum_area'].to_numpy())
        aggregated_energy_trigger.append(df_new.loc[df_new['triggerBits'] == 2, 'sum_area'].to_numpy())
        if cut_deltaT is not None and cut_totalcharge is not None:
            aggregated_cut_deltaT.append(cut_deltaT)
            aggregated_cut_sum_area.append(cut_totalcharge)
    
    # --- Aggregated Histograms ---
    total_folder = newcut_folder + "Total/"
    os.makedirs(total_folder, exist_ok=True)
    
    # Aggregated Trigger Bits Histogram.
    all_trig = np.concatenate(aggregated_trigger_bits) if aggregated_trigger_bits else np.array([])
    trig_bins = np.linspace(0, 35, 36)
    hist_trig, _ = np.histogram(all_trig, bins=trig_bins)
    trig_err = np.sqrt(hist_trig)
    plt.figure(figsize=(8, 5))
    plt.errorbar((trig_bins[:-1]+trig_bins[1:])/2, hist_trig, yerr=trig_err, fmt='o', label=f"Runs {start_run}-{end_run}")
    plt.xlabel("Trigger Bits")
    plt.ylabel("Events")
    if logscale:
        plt.yscale("log")
    plt.title("Aggregated Histogram of Trigger Bits")
    plt.legend()
    plt.tight_layout()
    trig_total_img = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_trigger_bits_histogram.png")
    plt.savefig(trig_total_img)
    plt.close()
    trig_total_pkl = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_trigger_bits_histogram.pkl")
    with open(trig_total_pkl, "wb") as f:
        pickle.dump({"hist": hist_trig, "bins": trig_bins}, f)
    
    # Aggregated Energy Histogram.
    all_energy = np.concatenate(aggregated_energy_all) if aggregated_energy_all else np.array([])
    trig_energy = np.concatenate(aggregated_energy_trigger) if aggregated_energy_trigger else np.array([])
    energy_bins = np.linspace(0, 100000, 101)
    plt.figure(figsize=(12, 8))
    plt.hist(all_energy, bins=energy_bins, alpha=0.7, edgecolor='black', label='All triggerBits')
    plt.hist(trig_energy, bins=energy_bins, alpha=0.7, edgecolor='black', label="triggerBits == 2")
    plt.xlabel("Total Charge (ADC)", fontsize=25)
    plt.ylabel("Events", fontsize=25)
    plt.title("Aggregated Histogram of Total Charge (ADC)", fontsize=25)
    if logscale:
        plt.yscale("log")
    plt.legend(fontsize=20)
    plt.tight_layout()
    energy_total_img = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_sum_area_histogram.jpg")
    plt.savefig(energy_total_img)
    plt.close()
    energy_total_pkl = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_sum_area_histogram.pkl")
    with open(energy_total_pkl, "wb") as f:
        pickle.dump({"all_data": all_energy, "trigger_data": trig_energy, "bins": energy_bins}, f)
    
    # Aggregated Cut Δt Histogram.
    if aggregated_cut_deltaT:
        all_deltaT = np.concatenate(aggregated_cut_deltaT)
        dt_bins_total = np.linspace(deltaT_cut[0], deltaT_cut[1], bins_num_cut+1)
        hist_dt, _ = np.histogram(all_deltaT, bins=dt_bins_total)
        dt_centers = (dt_bins_total[:-1] + dt_bins_total[1:]) / 2
        dt_err = np.sqrt(hist_dt)
        plt.figure(figsize=(10, 7))
        plt.errorbar(dt_centers, hist_dt, yerr=dt_err, fmt='o', label=f"Runs {start_run}-{end_run}: Δt cut {deltaT_cut[0]}-{deltaT_cut[1]} ns")
        plt.xlabel("Δt (ns)", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        plt.title("Aggregated Δt Histogram (Cut Events)", fontsize=16)
        if logscale:
            plt.yscale("log")
        plt.legend(fontsize=12)
        plt.tight_layout()
        dt_total_img = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_delta_t_histogram.png")
        plt.savefig(dt_total_img)
        plt.close()
        dt_total_pkl = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_delta_t_histogram.pkl")
        with open(dt_total_pkl, "wb") as f:
            pickle.dump({"hist": hist_dt, "bin_centers": dt_centers, "errorbars": dt_err}, f)
    
    # Aggregated Cut Total Charge Histogram.
    if aggregated_cut_sum_area:
        all_sum_area_cut = np.concatenate(aggregated_cut_sum_area)
        sa_bins_total = np.linspace(sum_area_cut[0], sum_area_cut[1], bins_num_cut+1)
        hist_sa, _ = np.histogram(all_sum_area_cut, bins=sa_bins_total)
        sa_centers = (sa_bins_total[:-1] + sa_bins_total[1:]) / 2
        sa_err = np.sqrt(hist_sa)
        plt.figure(figsize=(10, 7))
        plt.errorbar(sa_centers, hist_sa, yerr=sa_err, fmt='o', label=f"Runs {start_run}-{end_run}: sum_area cut {sum_area_cut[0]}-{sum_area_cut[1]} ADC")
        plt.xlabel("Total Charge (ADC)", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        plt.title("Aggregated Total Charge Histogram (Cut Events)", fontsize=16)
        if logscale:
            plt.yscale("log")
        plt.legend(fontsize=12)
        plt.tight_layout()
        sa_total_img = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_sum_area_histogram.png")
        plt.savefig(sa_total_img)
        plt.close()
        sa_total_pkl = os.path.join(total_folder, f"Total_run_{start_run}_{end_run}_sum_area_histogram.pkl")
        with open(sa_total_pkl, "wb") as f:
            pickle.dump({"hist": hist_sa, "bin_centers": sa_centers, "errorbars": sa_err}, f)
    
    print(f"\nAggregated histograms saved in: {total_folder}")

if __name__ == '__main__':
    main()
