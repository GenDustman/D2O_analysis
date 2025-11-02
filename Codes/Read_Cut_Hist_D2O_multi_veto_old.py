# #!/usr/bin/env python3
# """
# Refactored script for processing ROOT files with detailed configuration
# and an additional per-event time-std cut. Modular functions handle I/O,
# histogram plotting, Δt computation, and aggregated τ fitting.
# Includes low-light (triggerbit=16) analysis with multi-Gaussian fitting
# and new 3x3 correlation maps for key variables with correlation coefficients.

# MODIFICATION:
# - Changed analysis from total charge (ADC) to total photoelectrons (P.E.).
# - Error bars on P.E. histograms are calculated using simple Poisson counting
#   statistics (sqrt(N)), neglecting the uncertainty from the P.E. calculation itself.
# - ADDED: SiPM analysis for events with triggerbit >= 32, plotting area
#   histograms for channels 12-21.
# - ADDED: Aggregated "Total Photoelectron Comparison" plot.
# - REVISED: Veto Efficiency plot is now generated ONLY after all quality cuts
#   (multiplicity, P.E. range, time-std) have been applied.
# - REVISED: Thin veto panel (Ch 20, 21) analysis now combines data from both channels
#   and compares muon events (with coincidence) to all triggered events (no coincidence)
#   using normalized histograms.
# """
# import sys
# from pathlib import Path
# import pickle
# from datetime import datetime
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.colors import LogNorm
# from scipy.optimize import curve_fit
# from scipy.stats import pearsonr
# import uproot
# import awkward as ak
# import config

# def calculate_histograms(data_dict, config_dict):
#     """ Calculates histograms based on configuration. """
#     histograms = {}
#     edges = {}
#     for key, data in data_dict.items():
#         if key.endswith('_h'): # Height data
#             bins = np.linspace(*config_dict['height_range'], config_dict['height_bins'] + 1)
#             edges[key] = bins
#         elif key.endswith('_a'): # Area data
#             bins = np.linspace(*config_dict['area_range'], config_dict['area_bins'] + 1)
#             edges[key] = bins
#         else:
#             continue # Skip if not height or area

#         if data is not None and data.size > 0:
#             histograms[key], _ = np.histogram(data, bins=bins)
#         else:
#             histograms[key] = np.zeros(len(bins) - 1)
#             
#     return histograms, edges

# def ensure_dir(path: Path):
#     """
#     Ensure that a directory exists; create it and any parent directories if necessary.
#     """
#     path.mkdir(parents=True, exist_ok=True)


# def save_pickle(data: dict, path: Path):
#     """
#     Serialize and save a Python dictionary to a pickle file.
#     """
#     with path.open('wb') as f:
#         pickle.dump(data, f)

# def bin_edges_from_spec(bins_spec, data, data_range):
#     """
#     Return bin edges for np.histogram / plotting.
#     - bins_spec: int | str('auto','fd','scott',...) | array-like (edges)
#     - data: 1D array of samples (needed if bins_spec is a rule string)
#     - data_range: (lo, hi) numeric range to respect
#     """
#     data = np.asarray(data)
#     lo, hi = data_range
#     if isinstance(bins_spec, (int, np.integer)):
#         return np.linspace(lo, hi, int(bins_spec) + 1)
#     if isinstance(bins_spec, str):
#         # use NumPy’s selector to compute edges from rule
#         return np.histogram_bin_edges(data[np.isfinite(data)], bins=bins_spec, range=(lo, hi))
#     # allow passing explicit edges
#     edges = np.asarray(bins_spec)
#     if edges.ndim != 1 or edges.size < 2:
#         raise ValueError("bins_spec must be int, rule string, or 1D edges array")
#     return edges

# def make_dt_edges(delta_t_range):
#     """
#     Build Δt bin edges using a dedicated width aligned to the DAQ time tick.
#     Edges start at DELTA_T_LEFT_EDGE_NS and use a bin width that is an integer
#     multiple of TIME_TICK_NS to avoid aliasing patterns.
#     """
#     dt_min, dt_max = delta_t_range
#     tick = getattr(config, 'TIME_TICK_NS', 16)
#     width = getattr(config, 'DELTA_T_BIN_WIDTH_NS', tick)
#     left = getattr(config, 'DELTA_T_LEFT_EDGE_NS', 0)

#     # Force width and left edge to the tick grid
#     width = int(round(width / tick)) * tick
#     width = max(width, tick)
#     left = int(round(left / tick)) * tick

#     # Snap the plotting window to the tick grid
#     start = int(np.floor((dt_min - left) / tick)) * tick + left
#     stop  = int(np.ceil((dt_max - left) / tick)) * tick + left

#     # Build edges (ensure we cover the full [dt_min, dt_max] range)
#     edges = np.arange(start, stop + width, width)
#     return edges


# def plot_histogram(arrays, labels, bins, img_path, title, xlabel,
#                    M1_or_M2, logscale=True, figsize=(10, 6)):
#     """
#     Plot one or more datasets as overlapping histograms, with consistent styling.
#     Accepts `bins` as an int, rule string ('auto','fd','scott',...), or explicit edges.
#     Robust to empty arrays: computes a common edges array once up front.
#     """
#     plt.figure(figsize=figsize)
#     outputs = []

#     # Normalize `bins` to explicit EDGES once, using pooled non-empty data to set range.
#     nonempty = [a for a in arrays if a is not None and getattr(a, "size", 0) > 0]
#     if len(nonempty) == 0:
#         # Nothing to plot; make a dummy single-bin edge array
#         edges_final = np.array([0.0, 1.0])
#     else:
#         data_all = np.concatenate(nonempty)
#         lo, hi = np.nanmin(data_all), np.nanmax(data_all)
#         # Guard degenerate/invalid ranges
#         if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
#             lo, hi = 0.0, 1.0
#         # Convert int/'fd'/'auto'/edges -> explicit edges
#         if isinstance(bins, (int, np.integer, str)) or (np.asarray(bins).ndim != 1):
#             edges_final = bin_edges_from_spec(bins, data_all, (lo, hi))
#         else:
#             edges_final = np.asarray(bins)

#     for data, label in zip(arrays, labels):
#         if data is not None and getattr(data, "size", 0) > 0:
#             counts, edges, _ = plt.hist(
#                 data,
#                 bins=edges_final,
#                 alpha=0.7,
#                 edgecolor='black',
#                 label=f"{label} (N={len(data)})"
#             )
#             outputs.append((counts, edges))
#         else:
#             # Handle empty data case with precomputed edges
#             outputs.append((np.zeros(len(edges_final) - 1), edges_final))

#     plt.xlabel(xlabel)
#     plt.ylabel('Events')
#     plt.title(f"{title} ({M1_or_M2})")
#     if logscale:
#         plt.yscale('log')
#     plt.legend()
#     plt.minorticks_on()
#     plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
#     plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
#     plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
#     plt.tight_layout()
#     plt.savefig(img_path)

#     # Save histogram data as pickle
#     pkl_path = img_path.with_suffix('.pkl')
#     edges0 = outputs[0][1]
#     centers = 0.5 * (edges0[:-1] + edges0[1:])
#     pickle_data = {
#         'centers': centers,
#         'histograms': {label: counts for label, (counts, _) in zip(labels, outputs)}
#     }
#     save_pickle(pickle_data, pkl_path)
#     plt.close()
#     return outputs



# def plot_veto_efficiency(trig2_pe, trig2_or_34_pe, bins, vetorange, pe_range, img_path, pkl_path, title, M1_or_M2):
#     """
#     Calculates and plots veto efficiency as a function of total photoelectrons.
#     Efficiency = 1 - N(trig=2) / N(trig=2 or 34)
#     """
#     if trig2_or_34_pe.size == 0:
#         print(f"No events for veto efficiency calculation for {title}. Skipping.")
#         return

#     pe_min, pe_max = pe_range
#     bin_edges = np.linspace(pe_min, pe_max, bins + 1)
#     
#     counts_2, _ = np.histogram(trig2_pe, bins=bin_edges)
#     counts_2_or_34, _ = np.histogram(trig2_or_34_pe, bins=bin_edges)
#     
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # --- Calculate Efficiency and Error ---
#     efficiency = np.zeros_like(counts_2, dtype=float)
#     error = np.zeros_like(counts_2, dtype=float)
#     valid_mask = counts_2_or_34 > 0
#     
#     # Ratio p = k/n where k = counts_2 and n = counts_2_or_34
#     ratio = np.divide(counts_2[valid_mask], counts_2_or_34[valid_mask])
#     efficiency[valid_mask] = 1 - ratio
#     # Calculate average efficiency for valid bins within vetorange
#     average_efficiency = np.mean(efficiency[valid_mask & (bin_centers >= vetorange[0]) & (bin_centers <= vetorange[1])])
#     # Binomial error for the ratio p: sqrt(p * (1-p) / n)
#     n = counts_2_or_34[valid_mask]
#     p = ratio
#     error[valid_mask] = np.sqrt(p * (1 - p) / n)

#     # --- Plotting ---
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(bin_centers[valid_mask], efficiency[valid_mask], yerr=error[valid_mask],
#                  fmt='o', capsize=3, label='efficiency = 1 - N(trig=2) / N(trig=2 or 34)', color='navy', markersize=5)
#     plt.axhline(average_efficiency, color='red', linestyle='--',
#                 label=f'Average Efficiency = {average_efficiency:.4f}')
#     plt.xlabel('Total Photoelectrons (P.E.)')
#     plt.ylabel('Veto Efficiency')
#     plt.title(f"{title} ({M1_or_M2})")
#     plt.xlim(vetorange)
#     plt.ylim(0, 1.1)
#     plt.grid(which='major', linestyle='-', linewidth=0.7)
#     plt.grid(which='minor', linestyle=':', linewidth=0.5)
#     plt.minorticks_on()
#     plt.tight_layout()
#     plt.legend()
#     ensure_dir(img_path.parent)
#     plt.savefig(img_path)
#     plt.close()

#     # --- Save Data ---
#     pickle_data = {
#         'centers': bin_centers, 'efficiency': efficiency, 'error': error,
#         'counts_2': counts_2, 'counts_2_or_34': counts_2_or_34
#     }
#     save_pickle(pickle_data, pkl_path)
#     print(f"Veto efficiency plot saved to {img_path}")
#     print(f"Veto efficiency data saved to {pkl_path}")


# def plot_sipm_histograms(df, output_dir, label, M1_or_M2, hist_bins=100, hist_range=(-50, 1500)):
#     """
#     Selects events with trigger bit >= 32 and plots area histograms for SiPM channels 12-21.
#     Saves the histogram data to a pickle file.
#     """
#     sipm_events = df[df['triggerBits'] >= 32].copy()
#     if sipm_events.empty:
#         print(f"No SiPM events (triggerBits >= 32) found for {label}. Skipping SiPM histograms.")
#         return

#     area_data = np.array(sipm_events['area_array'].to_list())
#     
#     sipm_channels = range(12, 22)
#     fig, axes = plt.subplots(3, 4, figsize=(20, 15))
#     fig.suptitle(f'SiPM Channel Area (triggerBits>=32) - {label} ({M1_or_M2})', fontsize=16)
#     axes = axes.flatten()
#     sipm_hist_data = {}

#     for i, ch in enumerate(sipm_channels):
#         ax = axes[i]
#         if ch < area_data.shape[1]:
#             ch_data = area_data[:, ch]
#             counts, edges, _ = ax.hist(ch_data, bins=hist_bins, range=hist_range, histtype='step', linewidth=1.5, color='darkcyan')
#             sipm_hist_data[ch] = {'counts': counts, 'edges': edges}
#             ax.set_title(f'SiPM Channel {ch}')
#             ax.set_xlabel('Area (ADC)')
#             ax.set_ylabel('Events')
#             ax.grid(True, which='both', linestyle=':')
#             ax.set_yscale('log')
#         else:
#             ax.text(0.5, 0.5, f'Channel {ch}\nNot Available', ha='center', va='center', transform=ax.transAxes)
#             ax.set_axis_off()
#     
#     for i in range(len(sipm_channels), len(axes)):
#         axes[i].set_axis_off()

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     ensure_dir(output_dir)
#     
#     filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
#     base_filename = f'{filename_label}_{M1_or_M2}_sipm_area_histograms'
#     img_save_path = output_dir / f'{base_filename}.png'
#     pkl_save_path = output_dir / f'{base_filename}.pkl'
#     
#     plt.savefig(img_save_path)
#     save_pickle(sipm_hist_data, pkl_save_path)
#     print(f"SiPM histograms saved to {img_save_path}")
#     print(f"SiPM histogram data saved to {pkl_save_path}")
#     plt.close()

# # MODIFICATION START: New function for normalized histogram comparison
# def plot_normalized_histogram_comparison(array1, label1, array2, label2, bins, img_path, title, xlabel, M1_or_M2, figsize=(10, 6)):
#     """
#     Plots two datasets as overlapping, normalized histograms for shape comparison.
#     Uses a log scale on the y-axis.
#     """
#     plt.figure(figsize=figsize)
#     outputs = {}
#     
#     # Plot array1 if it has data (e.g., Muon Events)
#     if array1.size > 0:
#         counts1, edges1, _ = plt.hist(
#             array1, bins=bins, alpha=0.7, edgecolor='black',
#             label=f"{label1} (N={len(array1)})", density=True
#         )
#         outputs[label1] = (counts1, edges1)
#     
#     # Plot array2 if it has data (e.g., All Triggered Events)
#     if array2.size > 0:
#         counts2, edges2, _ = plt.hist(
#             array2, bins=bins, alpha=0.7, histtype='step', linewidth=2,
#             label=f"{label2} (N={len(array2)})", density=True
#         )
#         outputs[label2] = (counts2, edges2)

#     plt.xlabel(xlabel)
#     plt.ylabel('Normalized Events')
#     plt.title(f"{title} ({M1_or_M2})")
#     plt.yscale('log')
#     plt.legend()
#     plt.minorticks_on()
#     plt.grid(which='major', axis='y', linestyle='-', linewidth=0.75, color='gray')
#     plt.grid(which='minor', axis='y', linestyle=':', linewidth=0.5, color='gray')
#     plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')
#     plt.tight_layout()
#     plt.savefig(img_path)
#     
#     # Save histogram data as pickle
#     pkl_path = img_path.with_suffix('.pkl')
#     if outputs:
#         any_edges = next(iter(outputs.values()))[1]
#         centers = 0.5 * (any_edges[:-1] + any_edges[1:])
#         pickle_data = {'centers': centers, 'histograms': {label: counts for label, (counts, _) in outputs.items()}}
#         save_pickle(pickle_data, pkl_path)
#     
#     plt.close()


# def plot_thin_veto_performance(df, pulseh_array, area_array, output_dir, label, M1_or_M2, 
#                                channels_to_analyze, thin_veto_threshold, multiplicity_cut, hist_config):
#     """
#     Analyzes a given list of veto panel channels.
#     Compares muon events against all triggered events in those panels.
#     Muon events are defined by a panel trigger + triggerBit==34 + multiplicity cut.
#     """
#     if pulseh_array.size == 0:
#         print(f"No pulseH data for veto panel analysis for {label}. Skipping.")
#         return None, None, None, None

#     ensure_dir(output_dir)
#     filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")

#     # Check if the data contains the required channels
#     max_ch_idx = max(channels_to_analyze)
#     if pulseh_array.shape[1] <= max_ch_idx:
#         print(f"ERROR: Not enough channels in data (found {pulseh_array.shape[1]}) for this analysis. Skipping.")
#         return None, None, None, None

#     # --- Data Collection ---
#     # Create lists to hold data from all channels before combining
#     muon_h_list, muon_a_list = [], []
#     no_co_h_list, no_co_a_list = [], []

#     # Create a boolean mask for events that are muons according to the main detector
#     # This is calculated once and applied to all channels
#     pmt_muon_mask = (df['triggerBits'].values == 34) & (df['multiplicity'].values > multiplicity_cut)

#     # Loop over the specified channels to analyze
#     for ch in channels_to_analyze:
#         # Mask for events with a trigger in the current channel
#         triggered_this_ch = pulseh_array[:, ch] > thin_veto_threshold
#         
#         # Define the muon event for this channel
#         muon_mask_this_ch = triggered_this_ch & pmt_muon_mask
#         
#         # Extract and append data for muon events
#         muon_h_list.append(pulseh_array[muon_mask_this_ch, ch])
#         muon_a_list.append(area_array[muon_mask_this_ch, ch])

#         # Extract and append data for all triggered events (no coincidence)
#         no_co_h_list.append(pulseh_array[triggered_this_ch, ch])
#         no_co_a_list.append(area_array[triggered_this_ch, ch])
#     
#     # Combine the data from all channels into single arrays
#     combined_muon_h = np.concatenate(muon_h_list)
#     combined_muon_a = np.concatenate(muon_a_list)
#     combined_no_co_h = np.concatenate(no_co_h_list)
#     combined_no_co_a = np.concatenate(no_co_a_list)

#     # --- Plotting ---
#     # Plot Height Comparison
#     height_img_path = output_dir / f'{filename_label}_{M1_or_M2}_veto_panel_height_comparison.png'
#     plot_normalized_histogram_comparison(
#         array1=combined_muon_h, label1='Muon Events (Coincidence)',
#         array2=combined_no_co_h, label2='All Triggered Events',
#         bins=np.linspace(*hist_config['height_range'], hist_config['height_bins'] + 1),
#         img_path=height_img_path,
#         title=f'Veto Panel Height Comparison - {label}',
#         xlabel='Pulse Height (ADC)', M1_or_M2=M1_or_M2
#     )

#     # Plot Area Comparison
#     area_img_path = output_dir / f'{filename_label}_{M1_or_M2}_veto_panel_area_comparison.png'
#     plot_normalized_histogram_comparison(
#         array1=combined_muon_a, label1='Muon Events (Coincidence)',
#         array2=combined_no_co_a, label2='All Triggered Events',
#         bins=np.linspace(*hist_config['area_range'], hist_config['area_bins'] + 1),
#         img_path=area_img_path,
#         title=f'Veto Panel Area Comparison - {label}',
#         xlabel='Pulse Area (ADC)', M1_or_M2=M1_or_M2
#     )
#     
#     return combined_muon_h, combined_muon_a, combined_no_co_h, combined_no_co_a

# def compute_brn_data(df, pulseh_array, area_array, channels_to_analyze, brn_threshold):
#     """
#     Computes BRN delta_t and SiPM area data on a per-channel basis.
#     - BRN delta_t: Time between SiPM event (trig 32/34) and previous beam-on (trig 0).
#     - Data is stored only for channels that exceed the brn_threshold.
#     """
#     # 1. Get times for beam-on (trig 0) and SiPM (trig 32/34) events
#     beam_on_times = df.loc[df['triggerBits'] == 0, 'nsTime'].values
#     sipm_events = df.loc[(df['triggerBits'] == 32) | (df['triggerBits'] == 34)].copy()

#     if sipm_events.empty or beam_on_times.size == 0:
#         print("No SiPM events or no beam-on events. Skipping BRN analysis.")
#         return {}

#     sipm_times = sipm_events['nsTime'].values
#     
#     # 2. Compute BRN delta_t for all SiPM events
#     idx = np.searchsorted(beam_on_times, sipm_times, side='right')
#     delta_t = np.full(sipm_times.shape, np.nan)
#     valid = idx > 0
#     delta_t[valid] = sipm_times[valid] - beam_on_times[idx[valid] - 1]
#     sipm_events['brn_delta_t'] = delta_t
#     
#     # Get the corresponding pulseH and area arrays for the SiPM events
#     sipm_indices = sipm_events.index
#     sipm_pulseh_array = pulseh_array[sipm_indices]
#     sipm_area_array = area_array[sipm_indices]
#     all_brn_delta_t = sipm_events['brn_delta_t'].values

#     # 3. Initialize data structure
#     channel_data = {ch: {'delta_t': [], 'area': []} for ch in channels_to_analyze}

#     # 4. Populate delta_t data
#     for i in range(len(sipm_events)):
#         event_dt = all_brn_delta_t[i]
#         if not np.isfinite(event_dt):
#             continue  # Skip events with no preceding beam-on signal
#         
#         event_pulseh = sipm_pulseh_array[i]
#         for ch in channels_to_analyze:
#             if event_pulseh[ch] > brn_threshold:
#                 channel_data[ch]['delta_t'].append(event_dt)

#     # 5. Filter events by delta_t cut
#     dt_min, dt_max = config.BRN_DELTA_T_RANGE
#     dt_cut_mask = (sipm_events['brn_delta_t'] >= dt_min) & (sipm_events['brn_delta_t'] <= dt_max)
#     events_in_dt_range = sipm_events[dt_cut_mask]
#     
#     # 6. Populate area data (only for events in dt cut)
#     if not events_in_dt_range.empty:
#         filtered_indices = events_in_dt_range.index
#         # Get the original full-dataframe indices
#         original_indices_mask = np.isin(df.index, filtered_indices)
#         
#         # We need to get the pulseH/area arrays corresponding to the filtered events
#         # We can re-use the sipm_events dataframe which is already indexed correctly
#         filtered_pulseh_array = pulseh_array[sipm_indices][dt_cut_mask]
#         filtered_area_array = area_array[sipm_indices][dt_cut_mask]

#         for i in range(len(events_in_dt_range)):
#             event_pulseh = filtered_pulseh_array[i]
#             event_area = filtered_area_array[i]
#             for ch in channels_to_analyze:
#                 if event_pulseh[ch] > brn_threshold:
#                     channel_data[ch]['area'].append(event_area[ch])

#     # 7. Convert lists to numpy arrays for easier aggregation
#     for ch in channels_to_analyze:
#         channel_data[ch]['delta_t'] = np.array(channel_data[ch]['delta_t'])
#         channel_data[ch]['area'] = np.array(channel_data[ch]['area'])

#     return channel_data


# def plot_brn_histograms(channel_data, output_dir, label, M1_or_M2, 
#                         brn_dt_range, hist_config):
#     """
#     Plots the per-channel BRN delta_t and area histograms.
#     """
#     if not channel_data:
#         return # Skip if no data
#         
#     ensure_dir(output_dir)
#     filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
#     channels_to_analyze = list(channel_data.keys())
#     num_channels = len(channels_to_analyze)
#     
#     # --- 1. Plot BRN Delta T Histograms ---
#     fig_dt, axes_dt = plt.subplots(3, 4, figsize=(20, 15))
#     fig_dt.suptitle(f'BRN $\Delta t$ by Channel (Trig > {config.BRN_SIPM_THRESHOLD_ADC} ADC) - {label} ({M1_or_M2})', fontsize=16)
#     axes_dt = axes_dt.flatten()
#     
#     dt_min, dt_max = brn_dt_range
#     dt_bin_width = getattr(config, 'BRN_DELTA_T_BIN_WIDTH_NS', 128) 
#     dt_bins = np.arange(dt_min, dt_max + dt_bin_width, dt_bin_width)
#     
#     plot_data_dt = {}

#     for i, ch in enumerate(channels_to_analyze):
#         ax = axes_dt[i]
#         dt_data = channel_data[ch]['delta_t']
#         
#         if dt_data.size > 0:
#             counts, edges, _ = ax.hist(dt_data, bins=dt_bins, histtype='step', linewidth=1.5, color='darkgreen',
#                                        label=f"N = {dt_data.size}")
#             ax.set_yscale('log')
#             ax.legend()
#             plot_data_dt[ch] = {'counts': counts, 'edges': edges}
#         else:
#             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
#             
#         ax.set_title(f'Channel {ch}')
#         ax.set_xlabel('BRN $\Delta t$ (ns)')
#         ax.set_ylabel('Events')
#         ax.set_xlim(brn_dt_range)
#         ax.grid(True, which='both', linestyle=':')

#     # Turn off extra axes
#     for i in range(num_channels, len(axes_dt)):
#         axes_dt[i].set_axis_off()

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     dt_img_path = output_dir / f'{filename_label}_{M1_or_M2}_brn_delta_t.png'
#     plt.savefig(dt_img_path)
#     plt.close(fig_dt)

#     # --- 2. Plot BRN Area Histograms ---
#     fig_area, axes_area = plt.subplots(3, 4, figsize=(20, 15))
#     fig_area.suptitle(f'BRN SiPM Area (in $\Delta t$ cut) by Channel - {label} ({M1_or_M2})', fontsize=16)
#     axes_area = axes_area.flatten()

#     area_bins = np.linspace(*hist_config['area_range'], hist_config['area_bins'] + 1)
#     plot_data_area = {}

#     for i, ch in enumerate(channels_to_analyze):
#         ax = axes_area[i]
#         area_data = channel_data[ch]['area']
#         
#         if area_data.size > 0:
#             counts, edges, _ = ax.hist(area_data, bins=area_bins, histtype='step', linewidth=1.5, color='purple',
#                                        label=f"N = {area_data.size}")
#             ax.set_yscale('log')
#             ax.legend()
#             plot_data_area[ch] = {'counts': counts, 'edges': edges}
#         else:
#             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

#         ax.set_title(f'Channel {ch}')
#         ax.set_xlabel(f'Area (ADC) [$\Delta t$ in {brn_dt_range} ns]')
#         ax.set_ylabel('Events')
#         ax.set_xlim(hist_config['area_range'])
#         ax.grid(True, which='both', linestyle=':')

#     # Turn off extra axes
#     for i in range(num_channels, len(axes_area)):
#         axes_area[i].set_axis_off()

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     area_img_path = output_dir / f'{filename_label}_{M1_or_M2}_brn_area.png'
#     plt.savefig(area_img_path)
#     plt.close(fig_area)

#     # --- 3. Save pickled data ---
#     pkl_save_path = output_dir / f'{filename_label}_{M1_or_M2}_brn_analysis_data.pkl'
#     pickle_data = {
#         'channel_data': channel_data,
#         'dt_hist_data': plot_data_dt,
#         'area_hist_data': plot_data_area
#     }
#     save_pickle(pickle_data, pkl_save_path)
#     
#     print(f"BRN analysis plots saved to {dt_img_path} and {area_img_path}")
#     print(f"BRN analysis data saved to {pkl_save_path}")

# def plot_correlation_maps(df, output_dir, label, M1_or_M2):
#     """
#     Plots a 3x3 grid of correlation maps for delta_t, total_pe, and multiplicity.
#     """
#     ensure_dir(output_dir)
#     if df.empty:
#         print(f"DataFrame is empty for {label}. Skipping correlation map.")
#         return

#     variables = ['delta_t', 'total_pe', 'multiplicity']
#     pretty_labels = ['Δt (ns)', 'Total Photoelectrons', 'Multiplicity']

#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#     fig.suptitle(f'Correlation Matrix ({label}, {M1_or_M2})', fontsize=18)

#     for i in range(3):
#         for j in range(3):
#             ax = axes[i, j]
#             var_y = variables[i]
#             var_x = variables[j]

#             if i == 2: ax.set_xlabel(pretty_labels[j], fontsize=12)
#             if j == 0: ax.set_ylabel(pretty_labels[i], fontsize=12)

#             if i == j:
#                 data = df[var_x].dropna()
#                 if not data.empty:
#                     ax.hist(data, bins=50, histtype='step', linewidth=1.5, color='k')
#                 ax.set_yscale('log')
#                 ax.grid(True, which='both', linestyle=':')
#             else:
#                 subset = df[[var_x, var_y]].dropna()
#                 if not subset.empty and len(subset) > 1:
#                     h = ax.hist2d(subset[var_x], subset[var_y],
#                                   bins=50, cmap='viridis', norm=LogNorm())
#                     if h[0].max() > 0: fig.colorbar(h[3], ax=ax)

#                     corr, _ = pearsonr(subset[var_x], subset[var_y])
#                     corr_text = f'Corr: {corr:.2f}'
#                     ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=12,
#                             verticalalignment='top',
#                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#                 else:
#                     ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

#             if i < 2: ax.tick_params(axis='x', labelbottom=False)
#             if j > 0: ax.tick_params(axis='y', labelleft=False)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     
#     filename_label = label.replace(" ", "_").replace("-", "_").replace(":", "")
#     save_path = output_dir / f'{filename_label}_{M1_or_M2}_correlation_map.png'
#     
#     plt.savefig(save_path)
#     print(f"Correlation map saved to {save_path}")
#     plt.close()


# def compute_delta_t(df, muon_bits, veto_bits, mult_thresh):
#     """
#     Compute time differences Δt between veto events and the preceding muon event.
#     """
#     muon_mask = df['triggerBits'] >= muon_bits
#     veto_mask = (df['triggerBits'] == veto_bits) & (df['multiplicity'] > mult_thresh)
#     muon_times = df.loc[muon_mask, 'nsTime'].values
#     events = df.loc[veto_mask].copy()
#     times = events['nsTime'].values
#     idx = np.searchsorted(muon_times, times, side='right')
#     delta_t = np.full(times.shape, np.nan)
#     valid = idx > 0
#     delta_t[valid] = times[valid] - muon_times[idx[valid] - 1]
#     events['delta_t'] = delta_t
#     return events


# def save_cut_histograms(events, delta_t_range, pe_range, bins,
#                         save_dir, run_label, time_std_cut, M1_or_M2, logscale=True):
#     """
#     Apply sequential cuts and save errorbar histograms.
#     """
#     dt_min, dt_max = delta_t_range
#     pe_min, pe_max = pe_range

#     ensure_dir(save_dir)
#     sel = events.dropna(subset=['delta_t', 'total_pe']).copy()
#     print(f"{run_label}: after NaN drop: {len(sel)} events")
#     sel = sel[(sel['delta_t'] >= dt_min) & (sel['delta_t'] <= dt_max)]
#     print(f"{run_label}: after Δt cut: {len(sel)} events")
#     sel = sel[(sel['total_pe'] >= pe_min) & (sel['total_pe'] <= pe_max)]
#     print(f"{run_label}: after total_pe cut: {len(sel)} events")

#     sel = sel.dropna(subset=['time_std'])
#     sel = sel[sel['time_std'] < time_std_cut]
#     print(f"{run_label}: after time-std < {time_std_cut} ns cut: {len(sel)} events")

#     plot_correlation_maps(sel, save_dir, run_label, M1_or_M2)

#     if sel.empty:
#         return None, None, None

#     # --- Delta T Histogram ---
#     dt_bins = make_dt_edges((dt_min, dt_max))
#     dt_counts, dt_edges = np.histogram(sel['delta_t'], bins=dt_bins)
#     dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
#     dt_err = np.sqrt(dt_counts)
#     
#     dt_base_filename = f'delta_t_hist_{M1_or_M2}'
#     save_pickle({'hist': dt_counts, 'centers': dt_centers, 'errors': dt_err}, save_dir / f'{dt_base_filename}.pkl')
#     
#     plt.errorbar(dt_centers, dt_counts, yerr=dt_err, fmt='o', label=run_label)
#     plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'Δt Histogram ({M1_or_M2})')
#     if logscale: plt.yscale('log')
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.savefig(save_dir / f'{dt_base_filename}.png'); plt.close()

#     # --- Total PE Histogram ---
#     pe_bins = bin_edges_from_spec(bins, sel['total_pe'].values, (pe_min, pe_max))
#     pe_counts, pe_edges = np.histogram(sel['total_pe'], bins=pe_bins)
#     pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
#     peak_location = pe_centers[np.argmax(pe_counts)]
#     peak = np.round(peak_location, 1)
#     mean_pe = sel['total_pe'].mean()
#     mean_pe_val = np.round(mean_pe, 1)
#     pe_err = np.sqrt(pe_counts)

#     pe_base_filename = f'total_pe_hist_{M1_or_M2}'
#     save_pickle({'hist': pe_counts, 'centers': pe_centers, 'errors': pe_err}, save_dir / f'{pe_base_filename}.pkl')

#     plot_label = f'{run_label}\nMean = {mean_pe_val} p.e.'
#     plt.errorbar(pe_centers, pe_counts, yerr=pe_err, fmt='o', label=plot_label)
#     
#     plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title(f'Total Photoelectron Histogram ({M1_or_M2})')
#     plt.axvline(peak, color='red', linestyle='--', label=f'Peak = {peak} p.e.')
#     if logscale: plt.yscale('log')
#     plt.legend(); plt.grid(True); plt.tight_layout()
#     plt.savefig(save_dir / f'{pe_base_filename}.png'); plt.close()

#     return sel['delta_t'].values, sel['total_pe'].values, sel['multiplicity'].values



# def fit_and_plot_low_light(area_data, output_dir, file_label, M1_or_M2, hist_range, hist_bins=200):
#     """
#     Plots and fits sum_area for channels 0-11 for low-light events (triggerbit=16).
#     """
#     if area_data.size == 0:
#         print(f"No low-light data to process for {file_label}.")
#         return np.full(12, np.nan)

#     def constrained_gaussians(x, a0, mu0, sig0, a1, mu1, sig1, a2, a3):
#         sig2_sq = 2 * sig1**2 - sig0**2
#         sig3_sq = 3 * sig1**2 - 2 * sig0**2
#         if sig2_sq < 0 or sig3_sq < 0: return np.inf
#         pedestal = a0 * np.exp(-0.5 * ((x - mu0) / sig0)**2)
#         spe = a1 * np.exp(-0.5 * ((x - mu1) / sig1)**2)
#         dpe = a2 * np.exp(-0.5 * ((x - 2 * mu1) / np.sqrt(sig2_sq))**2)
#         tpe = a3 * np.exp(-0.5 * ((x - 3 * mu1) / np.sqrt(sig3_sq))**2)
#         return pedestal + spe + dpe + tpe

#     fig, axes = plt.subplots(3, 4, figsize=(20, 15))
#     fig.suptitle(f'Low-Light Channel Area Fits ({file_label}, {M1_or_M2})', fontsize=16)
#     axes = axes.flatten()
#     
#     mu1_values = np.full(12, np.nan)
#     fit_results_data = {}

#     for i in range(12):
#         ax = axes[i]
#         ch_data = area_data[:, i]
#         counts, edges = np.histogram(ch_data, bins=hist_bins, range=hist_range)
#         centers = 0.5 * (edges[:-1] + edges[1:])
#         ax.hist(ch_data, bins=edges, alpha=0.7, label=f'Ch {i} Data')

#         p0 = [counts.max(), 0, 20, counts.max()/5, 100, 30, counts.max()/25, counts.max()/125]
#         try:
#             mask = counts > 0
#             popt, pcov = curve_fit(constrained_gaussians, centers[mask], counts[mask], p0=p0, maxfev=10000)
#             perr = np.sqrt(np.diag(pcov))
#             mu1_values[i] = popt[4]
#             fit_x = np.linspace(hist_range[0], hist_range[1], 500)
#             ax.plot(fit_x, constrained_gaussians(fit_x, *popt), 'r-', label='Fit')
#             param_text = (f'$\\mu_1$: {popt[4]:.1f} ± {perr[4]:.1f}\n'
#                           f'$\\sigma_1$: {popt[5]:.1f} ± {perr[5]:.1f}')
#             ax.text(0.95, 0.95, param_text, transform=ax.transAxes, fontsize=9,
#                     verticalalignment='top', horizontalalignment='right',
#                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#             fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': popt, 'perr': perr}
#         except (RuntimeError, ValueError):
#             ax.text(0.5, 0.5, 'Fit Failed', transform=ax.transAxes, color='red', ha='center', va='center')
#             fit_results_data[i] = {'counts': counts, 'edges': edges, 'popt': None, 'perr': None}

#         ax.set_title(f'Channel {i}')
#         ax.set_xlabel('Sum Area (ADC)')
#         ax.set_ylabel('Events')
#         ax.grid(True, which='both', linestyle=':')
#         ax.legend(loc='lower left', fontsize='small')

#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     ensure_dir(output_dir)
#     
#     filename_label = file_label.replace(" ", "_").replace("-", "_").replace(":", "")
#     base_filename = f'{filename_label}_{M1_or_M2}_low_light_fits'
#     img_save_path = output_dir / f'{base_filename}.png'
#     pkl_save_path = output_dir / f'{base_filename}.pkl'
#     
#     plt.savefig(img_save_path)
#     save_pickle(fit_results_data, pkl_save_path)
#     print(f"Low-light fits saved to {img_save_path}")
#     print(f"Low-light fit data saved to {pkl_save_path}")
#     plt.close()
#     
#     return mu1_values


# def calculate_total_pe(df, mu1_values):
#     """
#     Calculates the total photoelectrons for each event using per-channel gain.
#     """
#     if np.all(np.isnan(mu1_values)):
#         print("ERROR: Low-light fit failed. Cannot calculate photoelectrons.")
#         return np.full(len(df), np.nan)

#     mu1_safe = np.where(np.isnan(mu1_values) | (mu1_values <= 0), np.inf, mu1_values)
#     if np.any(mu1_safe == np.inf):
#         nan_ch = np.where(np.isnan(mu1_values) | (mu1_values <= 0))[0]
#         print(f"Warning: mu1 fit failed/invalid for channels {nan_ch}. These channels will be excluded from the P.E. sum.")
#     
#     area_data_np = np.array(df['area_array'].to_list())[:, :12]
#     pe_per_channel = area_data_np / mu1_safe
#     total_pe = np.sum(pe_per_channel, axis=1)
#     
#     return total_pe


# def process_run(run, data_dir, output_dir, delta_t_cut, pe_cut, bins, veto_bins, vetorange,
#                 multiplicity_spe, multiplicity_cut, time_std_cut, logscale,
#                 low_light_fit_range, sipm_hist_config, M1_or_M2):
#     """
#     Process a single run: read data, perform calculations, and apply cuts.
#     """
#     print(f"--- Processing run {run} ---")
#     if M1_or_M2 == 'M1':
#         infile = data_dir / f"run{run}_processed_v5.root"
#     elif M1_or_M2 == 'M2':
#         infile = data_dir / f"run{run}_processed_H2O_v5.root"
#     if not infile.exists():
#         print(f"Missing file: {infile}")
#         return None

#     run_start_time_str = "no_ts"
#     try:
#         with uproot.open(infile) as f_ts:
#             if 'starttime' in f_ts:
#                 unix_time = f_ts['starttime'].member("fVal")
#                 run_start_time_str = datetime.fromtimestamp(unix_time).strftime('%Y%m%d-%H')
#     except Exception as e:
#         print(f"Warning: Could not read start time for run {run}, using default folder name. Error: {e}")

#     dfs = []
#     branches = ['eventID', 'nsTime', 'triggerBits', 'area', 'peakPosition', 'pulseH']
#     try:
#         for chunk in uproot.open(infile)['tree'].iterate(branches, library='ak', step_size='500 MB'):
#             df = pd.DataFrame({
#                 'eventID': ak.to_numpy(chunk['eventID']),
#                 'nsTime': ak.to_numpy(chunk['nsTime']),
#                 'triggerBits': ak.to_numpy(chunk['triggerBits']),
#                 'area_array': ak.to_list(chunk['area']),
#                 'peakPosition': ak.to_list(chunk['peakPosition']),
#                 'pulseH_array': ak.to_list(chunk['pulseH']),
#             })
#             dfs.append(df)
#     except uproot.KeyInFileError:
#         print(f"Warning: 'pulseH' branch not found in {infile}. Reading without it.")
#         dfs = []
#         branches.remove('pulseH')
#         for chunk in uproot.open(infile)['tree'].iterate(branches, library='ak', step_size='500 MB'):
#             df = pd.DataFrame({
#                 'eventID': ak.to_numpy(chunk['eventID']),
#                 'nsTime': ak.to_numpy(chunk['nsTime']),
#                 'triggerBits': ak.to_numpy(chunk['triggerBits']),
#                 'area_array': ak.to_list(chunk['area']),
#                 'peakPosition': ak.to_list(chunk['peakPosition']),
#             })
#             dfs.append(df)

#     if not dfs:
#         return None
#     df_all = pd.concat(dfs, ignore_index=True)

#     run_dir = output_dir / f"run{run}_{run_start_time_str}"
#     hist_dir = run_dir / "histograms"
#     cut_dir = run_dir / "cuthist"
#     ll_dir = run_dir / "lowlight"
#     ensure_dir(hist_dir); ensure_dir(cut_dir); ensure_dir(ll_dir)

#     plot_histogram([df_all['triggerBits'].to_numpy()], ['triggerBits'],
#                    np.arange(0, 37), hist_dir / f"{run}_{M1_or_M2}_triggerBits.png",
#                    f"Run {run} Trigger Bits", "Trigger Bits", M1_or_M2, logscale)
#     
#     ll_events = df_all[df_all['triggerBits'] == 16]
#     low_light_area_data = np.array(ll_events['area_array'].to_list())[:, :12] if not ll_events.empty else np.array([])

#     ll_hist_counts = {}
#     ll_bin_edges = np.linspace(*config.LOW_LIGHT_FIT_RANGE, 201) # 200 bins
#     if low_light_area_data.size > 0 and low_light_area_data.ndim == 2:
#         for i in range(12): # Channels 0-11
#              ch_data = low_light_area_data[:, i]
#              ll_hist_counts[i], _ = np.histogram(ch_data, bins=ll_bin_edges)
#     else:
#         # Create empty histograms if no data
#         for i in range(12):
#             ll_hist_counts[i] = np.zeros(len(ll_bin_edges) - 1)
#     
#     if low_light_area_data.size > 0:
#         mu1_values_run = fit_and_plot_low_light(low_light_area_data, ll_dir, f'Run{run}', M1_or_M2, hist_range=low_light_fit_range)
#     else:
#         print(f"No low-light events for run {run}. P.E. and multiplicity calculations will fail.")
#         mu1_values_run = np.full(12, np.nan)

#     area_data_np = np.array(df_all['area_array'].to_list())[:, :12]
#     times_data_np = np.array(df_all['peakPosition'].to_list())[:, :12]
#     mu1_safe = np.where(np.isnan(mu1_values_run) | (mu1_values_run <= 0), np.inf, mu1_values_run)
#     pe_per_channel = area_data_np / mu1_safe
#     postmcut_mask = pe_per_channel > multiplicity_spe
#     df_all['multiplicity'] = np.sum(postmcut_mask, axis=1)
#     masked_times = np.where(postmcut_mask, times_data_np, np.nan)
#     df_all['time_std'] = np.nanstd(masked_times, axis=1)

#     df_all['total_pe'] = calculate_total_pe(df_all, mu1_values_run)
#     df_all.to_pickle(run_dir / f"run{run}_{M1_or_M2}_data_with_pe.pkl")
#     
#     # ==============================================================================
#     # --- Apply event selection cuts and generate Veto Efficiency Plots ---
#     # ==============================================================================
#     print(f"Run {run}: Generating veto plots AFTER cuts...")
#     pe_min, pe_max = pe_cut
#     
#     passing_cuts_mask = (
#         (df_all['multiplicity'] > multiplicity_cut) &
#         (df_all['total_pe'] >= pe_min) & (df_all['total_pe'] <= pe_max) &
#         (df_all['time_std'] < time_std_cut)
#     )
#     df_filtered = df_all[passing_cuts_mask & df_all['total_pe'].notna()]
#     
#     pe_trig2 = df_filtered.loc[(df_filtered['triggerBits'] == 2), 'total_pe']
#     pe_trig2_or_34 = df_filtered.loc[(df_filtered['triggerBits'] == 2) | (df_filtered['triggerBits'] == 34), 'total_pe']
#     
#     # Build edges from the union of the two arrays to ensure consistent binning
#     pe_compare_data = np.concatenate(
#         [pe_trig2_or_34.values, pe_trig2.values]
#     ) if (len(pe_trig2_or_34) + len(pe_trig2)) > 0 else np.array([])
#     pe_compare_edges = bin_edges_from_spec(config.BINS, pe_compare_data, pe_cut)

#     plot_histogram(
#         [pe_trig2_or_34.values, pe_trig2.values],
#         ['Trig=2 or 34', 'Trig=2'],
#         pe_compare_edges,
#         hist_dir / f"{run}_{M1_or_M2}_total_pe_comparison.png",
#         'Total PE Comparison', 'Total P.E.', M1_or_M2, logscale
#     )

#     veto_img_path = hist_dir / f"{run}_{M1_or_M2}_veto_efficiency.png"
#     veto_pkl_path = hist_dir / f"{run}_{M1_or_M2}_veto_efficiency.pkl"
#     plot_veto_efficiency(pe_trig2.to_numpy(), pe_trig2_or_34.to_numpy(),
#                          veto_bins, vetorange, pe_cut, veto_img_path, veto_pkl_path,
#                          f"Veto Efficiency Run {run}", M1_or_M2)

#     # ==============================================================================
#     # --- Continue with Delta T analysis ---
#     # ==============================================================================
#     events = compute_delta_t(df_all, muon_bits=32, veto_bits=2, mult_thresh=multiplicity_cut)
#     cut_results = save_cut_histograms(
#         events, delta_t_cut, pe_cut, bins, cut_dir,
#         f"Run {run}", time_std_cut, M1_or_M2, logscale
#     )
#     
#     plot_sipm_histograms(df_all, run_dir, f"Run {run}", M1_or_M2, **sipm_hist_config)
#     
#     sipm_events_df = df_all[df_all['triggerBits'] >= 32]
#     
#     # MODIFICATION START: Call thin veto analysis function
#     tv_hist_counts = {} # Initialize empty dict for Thin Veto hists
#     tv_bin_edges = {}   # Initialize empty dict for Thin Veto edges
#     brn_data = {}       # Keep BRN data as is
#     
#     if config.PERFORM_THIN_VETO_ANALYSIS or config.PERFORM_BRN_ANALYSIS: # MODIFIED: Check for either analysis
#         if 'pulseH_array' in df_all.columns:
#             full_area_array = np.array(df_all['area_array'].to_list())
#             pulseh_array = np.array(df_all['pulseH_array'].to_list())
#             
#             # --- Thin Veto ---
#             if config.PERFORM_THIN_VETO_ANALYSIS:
#                 # Plotting function remains (optional, could remove per-run plot)
#                 tv_raw_data = plot_thin_veto_performance(
#                     df_all, pulseh_array, full_area_array, hist_dir, f"Run {run}", M1_or_M2,
#                     config.THIN_VETO_CHANNELS, config.THIN_VETO_THRESHOLD,
#                     config.MULTIPLICITY_CUT, config.THIN_VETO_HIST_CONFIG
#                 )
#                 # --- NEW: Calculate Thin Veto Histograms ---
#                 if tv_raw_data:
#                     tv_muon_h, tv_muon_a, tv_no_co_h, tv_no_co_a = tv_raw_data
#                     tv_data_dict = {
#                         'muon_h': tv_muon_h, 'muon_a': tv_muon_a,
#                         'no_co_h': tv_no_co_h, 'no_co_a': tv_no_co_a
#                     }
#                     tv_hist_counts, tv_bin_edges = calculate_histograms(tv_data_dict, config.THIN_VETO_HIST_CONFIG)
#                 else:
#                     # Create empty histograms if no data
#                     keys = ['muon_h', 'muon_a', 'no_co_h', 'no_co_a']
#                     tv_hist_counts, tv_bin_edges = calculate_histograms({k: None for k in keys}, config.THIN_VETO_HIST_CONFIG)
#             
#             # --- BRN Analysis ---
#             if config.PERFORM_BRN_ANALYSIS:
#                 brn_data = compute_brn_data(
#                     df_all, pulseh_array, full_area_array,
#                     config.BRN_SIPM_CHANNELS,
#                     config.BRN_SIPM_THRESHOLD_ADC
#                 )
#                 if brn_data:
#                     plot_brn_histograms(
#                         brn_data, hist_dir, f"Run {run}", M1_or_M2,
#                         config.BRN_DELTA_T_RANGE, config.BRN_HIST_CONFIG
#                     )
#                     
#         else:
#             print(f"Warning: 'pulseH_array' column not found for run {run}. Skipping thin veto and BRN analysis.")
#             # Create empty histograms if pulseH is missing
#             keys = ['muon_h', 'muon_a', 'no_co_h', 'no_co_a']
#             tv_hist_counts, tv_bin_edges = calculate_histograms({k: None for k in keys}, config.THIN_VETO_HIST_CONFIG)

#     # --- MODIFIED: Update return tuple ---
#     # Return dt_vals, pe_vals, mult_vals (as before)
#     # Return ll_hist_counts (NEW dict), ll_bin_edges (NEW array)
#     # Return sipm_events_df, pe_trig2, pe_trig2_or_34 (as before)
#     # Return tv_hist_counts (NEW dict), tv_bin_edges (NEW dict)
#     # Return brn_data (as before)
#     if cut_results:
#         dt_vals, pe_vals, mult_vals = cut_results
#         return (dt_vals, pe_vals, mult_vals,
#                 ll_hist_counts, ll_bin_edges, # NEW
#                 sipm_events_df, pe_trig2, pe_trig2_or_34,
#                 tv_hist_counts, tv_bin_edges, # NEW
#                 brn_data)
#     else:
#         # Need to return empty/None structures matching the above
#         return (None, None, None,
#                 ll_hist_counts, ll_bin_edges, # NEW (even if empty)
#                 sipm_events_df, pd.Series(dtype=float), pd.Series(dtype=float),
#                 tv_hist_counts, tv_bin_edges, # NEW (even if empty)
#                 brn_data)

# def aggregate_plots(aggregated, delta_t_cut, pe_cut, bins,
#                     fit_window, output_dir, M1_or_M2, label, logscale_dt, logscale_pe,
#                     perform_fit=True):
#     """
#     Generate aggregated histograms with simple Poisson errors.
#     """
#     ensure_dir(output_dir)
#     dt_min, dt_max = delta_t_cut
#     all_dt = np.concatenate(aggregated['delta_t']) if aggregated['delta_t'] else np.array([])
#     
#     filename_label = label.replace(" ", "_").replace("-", "_")

#     if all_dt.size:
#         dt_bins = make_dt_edges((dt_min, dt_max))
#         hist_dt, dt_edges = np.histogram(all_dt, bins=dt_bins)
#         dt_centers = 0.5 * (dt_edges[:-1] + dt_edges[1:])
#         dt_err = np.sqrt(hist_dt)

#         pickle_data = {'centers': dt_centers, 'hist': hist_dt, 'errors': dt_err}
#         plt.errorbar(dt_centers, hist_dt, yerr=dt_err, fmt='o', label='Data')

#         if perform_fit:
#             t_low, t_high = fit_window
#             mask = (dt_centers >= t_low) & (dt_centers <= t_high) & (hist_dt > 0)
#             if np.any(mask):
#                 fit_x = dt_centers[mask]
#                 fit_y = np.log(hist_dt[mask])
#                 try:
#                     (slope, intercept), cov = np.polyfit(fit_x, fit_y, 1, w=np.sqrt(hist_dt[mask]), cov=True)
#                     slope_err = np.sqrt(cov[0, 0])
#                     tau = -1.0 / slope
#                     tau_err = slope_err / (slope**2)
#                     fit_line = np.exp(intercept + slope * dt_centers)
#                     plt.plot(dt_centers, fit_line, '--', label=f'Fit τ={tau:.1f}±{tau_err:.1f} ns')
#                     plt.axvspan(t_low, t_high, color='gray', alpha=0.2, label='Fit Range')
#                     pickle_data['tau'] = tau
#                     pickle_data['tau_err'] = tau_err
#                 except (np.linalg.LinAlgError, ValueError) as e:
#                     print(f"Warning: Could not perform the fit. Error: {e}")
#             else:
#                 print("Warning: No data in the specified fit window. Skipping the fit.")

#         plt.xlabel('Δt (ns)'); plt.ylabel('Counts'); plt.title(f'{label} Δt ({M1_or_M2})')
#         if logscale_dt: plt.yscale('log')
#         plt.legend(); plt.grid(which='both'); plt.tight_layout()
#         
#         dt_base_filename = f'{filename_label}_{M1_or_M2}_delta_t'
#         plt.savefig(output_dir / f'{dt_base_filename}.png'); plt.close()
#         save_pickle(pickle_data, output_dir / f'{dt_base_filename}.pkl')

#     all_pe = np.concatenate(aggregated['total_pe']) if aggregated['total_pe'] else np.array([])
#     if all_pe.size:
#         pe_min, pe_max = pe_cut
#         pe_bins = bin_edges_from_spec(bins, all_pe, (pe_min, pe_max))
#         hist_pe, pe_edges = np.histogram(all_pe, bins=pe_bins)
#         pe_centers = 0.5 * (pe_edges[:-1] + pe_edges[1:])
#         peak_location = pe_centers[np.argmax(hist_pe)]
#         peak = np.round(peak_location, 1)
#         mean_pe = all_pe.mean()
#         mean_pe_val = np.round(mean_pe, 1)
#         pe_err = np.sqrt(hist_pe)

#         plot_label = f'{label}\nMean = {mean_pe_val} p.e.'
#         plt.errorbar(pe_centers, hist_pe, yerr=pe_err, fmt='o', label=plot_label)
#         
#         plt.axvline(peak, color='red', linestyle='--', label=f'Peak = {peak} p.e.')
#         plt.xlabel('Total Photoelectrons'); plt.ylabel('Counts'); plt.title(f'{label} Total Photoelectrons ({M1_or_M2})')
#         if logscale_pe: plt.yscale('log')
#         plt.legend(); plt.grid(which='both'); plt.tight_layout()
#         
#         pe_base_filename = f'{filename_label}_{M1_or_M2}_total_pe'
#         plt.savefig(output_dir / f'{pe_base_filename}.png'); plt.close()
#         save_pickle({'centers': pe_centers, 'hist': hist_pe, 'errors': pe_err},
#                     output_dir / f'{pe_base_filename}.pkl')


# def main():
#     """
#     Entry point for a single sub-job. This script is called by SLURM.
#     It processes a range of runs and saves aggregated results for the master script.
#     """
#     if len(sys.argv) != 5:
#         print("Usage: python Read_Cut_Hist_D2O_multi.py <start_run> <end_run> <M1_or_M2> <top_output_dir>")
#         sys.exit(1)
#         
#     start_run = int(sys.argv[1])
#     end_run = int(sys.argv[2])
#     M1_or_M2 = sys.argv[3]
#     top_output_dir = Path(sys.argv[4])
#     
#     if M1_or_M2 == 'M1':
#         data_dir = Path(config.DATA_DIR_M1)
#     elif M1_or_M2 == 'M2':
#         data_dir = Path(config.DATA_DIR_M2)
#     else:
#         raise ValueError("M1_or_M2 must be 'M1' or 'M2'.")

#     output_dir = top_output_dir / f"subjob_{start_run}-{end_run}"
#     ensure_dir(output_dir)

#     print("=== Configuration ===")
#     print(f"Analysis type: {M1_or_M2}")
#     print(f"Runs: {start_run} to {end_run}")
#     print(f"Output Directory for this job: {output_dir}")
#     print(f"Δt cut: {config.DELTA_T_CUT} ns")
#     print(f"Photoelectron cut: {config.PE_CUT} P.E.")
#     print(f"Time-std cut: < {config.TIME_STD_CUT} ns")
#     if hasattr(config, 'PERFORM_THIN_VETO_ANALYSIS') and config.PERFORM_THIN_VETO_ANALYSIS:
#         print("Thin Veto Analysis: ENABLED")
#     else:
#         print("Thin Veto Analysis: DISABLED")
#         
#     # NEW: Print status for BRN analysis
#     if hasattr(config, 'PERFORM_BRN_ANALYSIS') and config.PERFORM_BRN_ANALYSIS:
#         print("BRN Analysis: ENABLED")
#     else:
#         print("BRN Analysis: DISABLED")
#         
#     print("======================")

#     # MODIFICATION START: Update aggregated dictionary
#     aggregated = {
#         'delta_t': [], 'total_pe': [], 'multiplicity': [],
#         # 'low_light_areas': [], # REMOVED
#         'sipm_events': [],
#         'pe_trig2': [], 'pe_trig2_or_34': [],
#         # 'thin_veto_muon_h': [], 'thin_veto_muon_a': [], # REMOVED
#         # 'thin_veto_no_co_h': [], 'thin_veto_no_co_a': [], # REMOVED
#         'brn_channel_data': [],
#         'low_light_hists': [], # NEW list to store dicts of counts per run
#         'thin_veto_hists': []  # NEW list to store dicts of counts per run
#     }
#     ll_bin_edges_agg = None # Store LL bin edges once
#     tv_bin_edges_agg = None # Store TV bin edges once

#     for run in range(start_run, end_run + 1):
#         result = process_run(
#             # ... (arguments as before) ...
#         )
#         
#         # --- MODIFIED: Unpack new results and append ---
#         if result:
#             (dt_vals, pe_vals, mult_vals,
#              ll_hists, ll_edges, # NEW
#              sipm_df, pe_2, pe_2_or_34,
#              tv_hists, tv_edges, # NEW
#              brn_data) = result
#             
#             if dt_vals is not None: aggregated['delta_t'].append(dt_vals)
#             if pe_vals is not None: aggregated['total_pe'].append(pe_vals)
#             if mult_vals is not None: aggregated['multiplicity'].append(mult_vals)
#             
#             # Store histograms
#             aggregated['low_light_hists'].append(ll_hists)
#             if ll_bin_edges_agg is None: ll_bin_edges_agg = ll_edges # Store edges from first valid run
#             
#             if not sipm_df.empty: aggregated['sipm_events'].append(sipm_df)
#             if not pe_2.empty: aggregated['pe_trig2'].append(pe_2)
#             if not pe_2_or_34.empty: aggregated['pe_trig2_or_34'].append(pe_2_or_34)

#             # Store histograms
#             aggregated['thin_veto_hists'].append(tv_hists)
#             if tv_bin_edges_agg is None: tv_bin_edges_agg = tv_edges # Store edges from first valid run

#             if brn_data: aggregated['brn_channel_data'].append(brn_data)
#     
#     agg_label = f"Runs {start_run}-{end_run}"

#     # --- Per-job aggregated plots (for diagnostics) ---
#     aggregate_plots(
#         aggregated, config.DELTA_T_CUT, config.PE_CUT, config.BINS, config.TAU_FIT_WINDOW,
#         output_dir, M1_or_M2, agg_label, config.LOGSCALE_DT_AGG, config.LOGSCALE_PE_AGG, perform_fit=config.DO_TAU_FIT
#     )
#     
#     if aggregated['pe_trig2_or_34']:
#         agg_pe_trig2 = pd.concat(aggregated['pe_trig2'], ignore_index=True)
#         agg_pe_trig2_or_34 = pd.concat(aggregated['pe_trig2_or_34'], ignore_index=True)
#         filename_label = agg_label.replace(" ", "_").replace("-", "_")
#         
#         veto_img_path = output_dir / f"{filename_label}_{M1_or_M2}_veto_efficiency_agg.png"
#         veto_pkl_path = output_dir / f"{filename_label}_{M1_or_M2}_veto_efficiency_agg.pkl"
#         plot_veto_efficiency(
#             agg_pe_trig2.to_numpy(), agg_pe_trig2_or_34.to_numpy(),
#             config.VETO_BINS, config.VETO_RANGE, config.PE_CUT, veto_img_path, veto_pkl_path,
#             f"Aggregated Veto Efficiency {agg_label}", M1_or_M2
#         )

#     # MODIFICATION START: Plot aggregated thin veto comparison histograms
#     # if config.PERFORM_THIN_VETO_ANALYSIS:
#     #     print("--- Generating Aggregated Thin Veto Comparison Plots ---")
#     #     filename_label = agg_label.replace(" ", "_").replace("-", "_")
#         
#     #     agg_muon_h = np.concatenate(aggregated['thin_veto_muon_h']) if aggregated['thin_veto_muon_h'] else np.array([])
#     #     agg_no_co_h = np.concatenate(aggregated['thin_veto_no_co_h']) if aggregated['thin_veto_no_co_h'] else np.array([])
#     #     agg_muon_a = np.concatenate(aggregated['thin_veto_muon_a']) if aggregated['thin_veto_muon_a'] else np.array([])
#     #     agg_no_co_a = np.concatenate(aggregated['thin_veto_no_co_a']) if aggregated['thin_veto_no_co_a'] else np.array([])
#         
#     #     # Plot aggregated height comparison
#     #     height_img_path = output_dir / f'{filename_label}_{M1_or_M2}_thin_veto_height_comparison_agg.png'
#     #     plot_normalized_histogram_comparison(
#     #         array1=agg_muon_h, label1='Muon Events (Coincidence)',
#     #         array2=agg_no_co_h, label2='All Triggered Events',
#     #         bins=np.linspace(*config.THIN_VETO_HIST_CONFIG['height_range'], config.THIN_VETO_HIST_CONFIG['height_bins'] + 1),
#     #         img_path=height_img_path, title=f'Agg. Thin Veto Height Comparison - {agg_label}',
#     #         xlabel='Pulse Height (ADC)', M1_or_M2=M1_or_M2
#     #     )
#     #     # Plot aggregated area comparison
#     #     area_img_path = output_dir / f'{filename_label}_{M1_or_M2}_thin_veto_area_comparison_agg.png'
#     #     plot_normalized_histogram_comparison(
#     #         array1=agg_muon_a, label1='Muon Events (Coincidence)',
#     #         array2=agg_no_co_a, label2='All Triggered Events',
#     #         bins=np.linspace(*config.THIN_VETO_HIST_CONFIG['area_range'], config.THIN_VETO_HIST_CONFIG['area_bins'] + 1),
#     #         img_path=area_img_path, title=f'Agg. Thin Veto Area Comparison - {agg_label}',
#     #         xlabel='Pulse Area (ADC)', M1_or_M2=M1_or_M2
#     #     )
#     # MODIFICATION END

#     # --- SAVE AGGREGATED DATA FOR MASTER SCRIPT ---
#     print(f"Saving aggregated data for sub-job {start_run}-{end_run}...")
#     try:
#         if aggregated['delta_t']:
#             np.save(output_dir / 'aggregated_delta_t.npy', np.concatenate(aggregated['delta_t']))
#             np.save(output_dir / 'aggregated_total_pe.npy', np.concatenate(aggregated['total_pe']))
#             np.save(output_dir / 'aggregated_multiplicity.npy', np.concatenate(aggregated['multiplicity']))

#         # if aggregated['low_light_areas']:
#         #     np.save(output_dir / 'aggregated_low_light_areas.npy', np.concatenate(aggregated['low_light_areas'], axis=0))

#         if aggregated['sipm_events']:
#              job_sipm_df = pd.concat(aggregated['sipm_events'], ignore_index=True)
#              job_sipm_area_data = job_sipm_df['area_array']
#              job_sipm_area_data.to_pickle(output_dir / 'aggregated_sipm_area_array.pkl')
#         if aggregated['pe_trig2']:
#              pd.concat(aggregated['pe_trig2'], ignore_index=True).to_pickle(output_dir / 'aggregated_pe_trig2.pkl')
#         if aggregated['pe_trig2_or_34']:
#              pd.concat(aggregated['pe_trig2_or_34'], ignore_index=True).to_pickle(output_dir / 'aggregated_pe_trig2_or_34.pkl')
#         
#         # --- NEW: Sum and Save Low-Light Histograms ---
#         if aggregated['low_light_hists'] and ll_bin_edges_agg is not None:
#              job_ll_master_counts = {ch: np.zeros_like(aggregated['low_light_hists'][0][ch]) for ch in range(12)}
#              for run_hists in aggregated['low_light_hists']:
#                  for ch in range(12):
#                      job_ll_master_counts[ch] += run_hists.get(ch, 0) # Sum counts safely
#              ll_save_data = {'counts': job_ll_master_counts, 'edges': ll_bin_edges_agg}
#              save_pickle(ll_save_data, output_dir / 'aggregated_low_light_hists.pkl')

#         # --- NEW: Sum and Save Thin Veto Histograms ---
#         if aggregated['thin_veto_hists'] and tv_bin_edges_agg is not None:
#              hist_keys = aggregated['thin_veto_hists'][0].keys()
#              job_tv_master_counts = {k: np.zeros_like(aggregated['thin_veto_hists'][0][k]) for k in hist_keys}
#              for run_hists in aggregated['thin_veto_hists']:
#                  for k in hist_keys:
#                       job_tv_master_counts[k] += run_hists.get(k, 0) # Sum counts safely
#              tv_save_data = {'counts': job_tv_master_counts, 'edges': tv_bin_edges_agg}
#              save_pickle(tv_save_data, output_dir / 'aggregated_thin_veto_hists.pkl')
#         
#         # NEW: Save aggregated BRN data
#         if config.PERFORM_BRN_ANALYSIS:
#             if aggregated['brn_channel_data']:
#                 with open(output_dir / 'aggregated_brn_channel_data.pkl', 'wb') as f:
#                     pickle.dump(aggregated['brn_channel_data'], f)
#         # MODIFICATION END
#             
#         print("Successfully saved data for master aggregation.")

#     except Exception as e:
#         print(f"An error occurred while saving aggregated data: {e}")

#     print("--- Sub-job Analysis Complete ---")
#     
# if __name__ == "__main__":
#     main()