# LUPEAPP Processed Dataset

This directory contains processed output files from the LUPE-AMPS analysis pipeline.

## ⚠️ Data Files Stored Externally

Due to file size constraints, the processed data files are hosted on Box rather than in this repository.

**📦 Download the data files here:** [LUPEAPP Processed Dataset (Box)](https://upenn.box.com/s/v3znq5d6373fx65pkw7u1ielkwe3na6w)

## Dataset Contents

### demo_FormalinResponse

| File | Description |
|------|-------------|
| `figures/` | Generated visualization outputs |
| `behaviors_demo_FormalinResponse.pkl` | Extracted behavior classifications |
| `binned_features_demo_FormalinResponse.pkl` | Time-binned behavioral features |
| `project_info_demo_FormalinResponse.txt` | Project metadata and parameters |
| `raw_data_demo_FormalinResponse.pkl` | Raw processed data |

### demo_LUPEAMPS_FormalinMorphine

| File | Description |
|------|-------------|
| `figures/` | Generated visualization outputs |
| `behaviors_demo_LUPEAMPS_FormalinMorphine.pkl` | Extracted behavior classifications |
| `binned_features_demo_LUPEAMPS_FormalinMorphine.pkl` | Time-binned behavioral features |
| `project_info_demo_LUPEAMPS_FormalinMorphine.txt` | Project metadata and parameters |
| `raw_data_demo_LUPEAMPS_FormalinMorphine.pkl` | Raw processed data |

## Usage

1. Download the dataset folder from the Box link above
2. Place the contents in this directory (`LUPEAPP_processed_dataset/`)
3. The analysis notebooks will automatically detect and load the files

## File Descriptions

- **behaviors_*.pkl** - Pickle files containing frame-by-frame behavior labels
- **binned_features_*.pkl** - Behavioral features aggregated into time bins for analysis
- **project_info_*.txt** - Text files with experimental parameters and metadata
- **raw_data_*.pkl** - Preprocessed data ready for LUPE-AMPS analysis
- **figures/** - Output visualizations including behavior distributions, transition matrices, and pain scores
