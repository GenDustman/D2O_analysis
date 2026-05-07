# D2O Analysis

[![GitHub](https://img.shields.io/badge/GitHub-GenDustman%2FD2O__analysis-blue)](https://github.com/GenDustman/D2O_analysis)

Analysis pipeline for the **COHERENT experiment's D2O (heavy water) detector** at the Spallation Neutron Source (SNS), Oak Ridge National Laboratory. The pipeline processes ROOT data from two liquid-scintillator detector modules to search for coherent elastic neutrino-nucleus scattering (CEvNS), measure muon lifetimes, study Michel electron spectra, and characterize detector response.

## Repository Structure

```
D2O_analysis/
├── Codes/                  # Main analysis code
│   ├── config.py           # Central configuration (active)
│   ├── Read_Cut_Hist_D2O_multi_veto.py   # Processing script (active)
│   ├── aggregate_master_veto.py          # Aggregation script (active)
│   ├── Globus_transfer.py                # Globus data transfer
│   ├── *.ipynb             # Jupyter notebooks (exploratory)
│   └── Read_Cut_Hist_*.py  # Older processing variants (legacy)
├── Scripts/                # Job submission and local execution
│   ├── submit_veto.sh      # Submit analysis jobs via SLURM (active)
│   ├── run_local.sh        # Run analysis locally without SLURM
│   ├── submit_veto_legacy.sh
│   ├── submit_aggregation_only.sh
│   ├── run_legacy_analysis.sh
│   └── test_env.sh
└── exercise/               # Tutorial workspace
    └── Read_Cut_Hist.ipynb
```

## Active vs Legacy Files

The repo contains multiple generations of scripts as the analysis evolved. Below is a guide to what should be used now.

### Active (current workflow)

| File | Purpose |
|---|---|
| `Codes/config.py` | Single source of truth — all cuts, channel maps, fit settings, directory paths |
| `Codes/Read_Cut_Hist_D2O_multi_veto.py` | SLURM sub-job processing: reads ROOT files, applies cuts, computes histograms, saves `.npy`/`.pkl` |
| `Codes/aggregate_master_veto.py` | Master aggregation: collects sub-job outputs, produces grand-aggregated plots and fits |
| `Codes/Globus_transfer.py` | Transfer ROOT files from ORNL to CMU via Globus |
| `Scripts/submit_veto.sh` | SLURM submission wrapper for the full pipeline |
| `Scripts/run_local.sh` | Local execution (no SLURM) — edit params and run directly |
| `Scripts/submit_aggregation_only.sh` | Re-run only the aggregation step on existing results |

### Legacy (superseded — kept for reference and older ROOT formats)

| File | Why it's legacy |
|---|---|
| `Codes/config_legacy.py` | Configuration for older ROOT v3 file format |
| `Codes/Read_Cut_Hist.py` | Very first processing script, hardcoded config |
| `Codes/Read_Cut_Hist_old.py` | Early version, limited features |
| `Codes/Read_Cut_Hist_D2O.py` | Second generation, no SLURM support |
| `Codes/Read_Cut_Hist_D2O_multi.py` | Third generation, no veto/BRN/Event61 analysis |
| `Codes/Read_Cut_Hist_spe.py` | SPE-only variant |
| `Codes/Read_Cut_Hist_D2O_multi_veto_old.py` | Earlier veto script, mostly commented out |
| `Codes/Read_Cut_Hist_D2O_multi_veto_legacy.py` | Veto analysis for v3 ROOT files |
| `Codes/aggregate_master.py` | Original aggregation, no veto/BRN/Event61 |
| `Codes/aggregate_master_veto_legacy.py` | Aggregation for legacy v3 data |
| `Scripts/submit.sh` | Original submission (no veto) |
| `Scripts/submit_veto_legacy.sh` | Submission for legacy v3 ROOT files |
| `Scripts/run_legacy_analysis.sh` | Direct (non-SLURM) legacy run |

The legacy scripts exist because: (1) the ROOT file format changed from v3 to v4/v5, requiring different branch handling, and (2) the analysis grew from simple muon-lifetime fits to include veto efficiency, BRN, thin-veto, Event61, and highlight-PE analyses.

## Analysis Pipeline

The analysis runs in three stages, orchestrated via SLURM on a cluster.

### Stage A: Data Transfer

Raw or processed ROOT files are transferred from ORNL's data servers to the local `/raid1/genli/Data_D2O/` storage using Globus.

```
python Codes/Globus_transfer.py
```

Data lands in `M1_data/` (Module 1) or `M2_data/` (Module 2), organized by run number.

### Stage B: Parallel Processing

`Scripts/submit_veto.sh` splits a range of runs across multiple SLURM array jobs. Each sub-job runs `Read_Cut_Hist_D2O_multi_veto.py`, which:

1. Reads ROOT trees via **uproot** + **awkward** in chunks
2. Converts ADC to photoelectrons (PE)
3. Applies quality cuts: time-std, PE range, PMT multiplicity, delta-t window
4. Identifies muon events (triggerBit ≥ 34) and veto events (triggerBit = 2)
5. Computes Δt between veto candidates and preceding muons
6. Fills histograms: Δt, total PE, SiPM pulse-height, SiPM area/height noise ratio, thin-veto, BRN, Event61
7. Performs low-light (triggerBit = 16) multi-Gaussian SPE fitting
8. Saves per-subjob `.npy` and `.pkl` outputs

```
sbatch Scripts/submit_veto.sh
```

### Stage C: Master Aggregation

When all sub-jobs complete, a dependent job runs `aggregate_master_veto.py`, which:

- Collects all sub-job outputs via memory-efficient incremental aggregation
- Produces grand-aggregated plots in a `MASTER_RESULTS/` directory:
  - **ΔT histogram** with muon lifetime (τ) exponential fit
  - **Total PE spectrum** with peak finding
  - **Veto efficiency** vs total PE
  - **Michel peak** position evolution over runs
  - **Low-light SPE** multi-Gaussian fits per SiPM panel
  - **BRN analysis**: Δt, area, and 2D heatmaps
  - **Event61** ADC spectrum and signal evolution
  - **Highlight PE** analysis with summed spectrum fitting
  - **SiPM pulse-height** Landau fits per channel
  - **SiPM noise ratio** — area/pulseHeight vs channel, for separating electronic noise from real pulses
  - **Thin-veto** panel comparisons

## Quick Start

### Prerequisites

- Python 3.11 conda environment: `/raid1/genli/conda/miniconda3/envs/py311/`
- Key packages: `uproot`, `awkward`, `numpy`, `scipy`, `matplotlib`, `pandas`
- SLURM cluster access (for batch processing)
- Globus CLI (for data transfer)
- Input data at `/raid1/genli/Data_D2O/M1_data/` and `/raid1/genli/Data_D2O/M2_data/`

### Running the analysis

1. **Configure** — Edit `Codes/config.py` to set:
   - `DATA_DIR_M1` / `DATA_DIR_M2` — input data paths
   - `suffix_M1` / `suffix_M2` — ROOT file suffix (e.g., `_processed_v5.root`)
   - Cuts: `PE_CUT`, `DELTA_T_CUT`, `MULTIPLICITY_CUT`, etc.
   - Toggles: `PERFORM_THIN_VETO_ANALYSIS`, `PERFORM_BRN_ANALYSIS`

2. **Submit (cluster)** — Adjust run range and number of jobs in `Scripts/submit_veto.sh`, then:
   ```bash
   sbatch Scripts/submit_veto.sh
   ```

3. **Run locally (no SLURM)** — Edit the parameters at the top of `Scripts/run_local.sh`, then:
   ```bash
   bash Scripts/run_local.sh
   ```

4. **Re-aggregate only** (if aggregation fails or config changes):
   ```bash
   sbatch Scripts/submit_aggregation_only.sh
   ```

### Running legacy analysis (older ROOT v3 files)

```bash
bash Scripts/run_legacy_analysis.sh
```

## Detector Configuration

The detector has two liquid-scintillator modules (M1 and M2) with D2O-loaded target volume:

| Component | Channels | Purpose |
|---|---|---|
| PMTs | 0–11 (12 channels) | Main light collection |
| SiPMs | 12–21 (10 channels) | Veto panels and thin-veto |
| Event61 | 22 (synthetic) | External scintillator for BRN cross-check |

DAQ time granularity is 16 ns. Muon lifetime for fitting is 2197 ns.

## Physics Analyses

- **Muon lifetime (τ)** — Exponential fit to Δt distribution between stopped muons and decay electrons
- **Veto efficiency** — SiPM rejection power as a function of total photoelectrons
- **Michel electron spectrum** — Endpoint energy and peak evolution over run periods
- **Single photoelectron (SPE) calibration** — Multi-Gaussian fits to low-light (triggerBit 16) events
- **Beam-related neutrons (BRN)** — Timing and energy spectra in a dedicated Δt window (832–4992 ns)
- **Thin-veto performance** — Muon-veto coincidence vs all-event comparisons
- **SiPM noise ratio** — Area/PulseHeight for triggerBits ≥ 32 muon events, per SiPM channel. The pulse-shape ratio helps discriminate real SiPM pulses from electronic noise spikes that could falsely cross the DAQ threshold (30 ADC)
- **Highlight PE analysis** — PMT charge response to high-energy deposition events

## Notebooks

Jupyter notebooks in `Codes/` are for exploratory and diagnostic work — they are **not** part of the batch pipeline:

| Notebook | Purpose |
|---|---|
| `MichelElectron.ipynb` | Michel electron spectrum fitting (two-region) |
| `Hist_analysis.ipynb` | General histogram exploration |
| `Spectrum_subtraction.ipynb` | Background subtraction analysis |
| `veto_efficiency.ipynb` | Veto efficiency studies |
| `DeltaT_Hist_analysis_test.ipynb` | Delta-T histogram testing |
| `Event61_debug.ipynb` | Event61 synthetic bit debugging |
| `D2O_legacy_test.ipynb` | Legacy data testing |
| `Globus_transfer.ipynb` | Interactive Globus transfers |
| `tools_runnum_run_time_convt.ipynb` | Run number ↔ timestamp conversion |
| `test.ipynb` | Scratch/testing |
