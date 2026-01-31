import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.join(os.path.abspath(''), '..') not in sys.path:
    sys.path.append(os.path.join(os.path.abspath(''), '..'))

from utils.classification import load_behaviors
from utils.meta import *


def behavior_binned_mouse_screening(project_name, output_analysis_dir=None, heatmap_max_value=None):
    """
    Perform behavioral screening analysis for each mouse using per-frame CSV files.

    This function loads CSV files with behavior classifications, processes the data to compute
    the number of frames per minute for each behavior per mouse, and then generates heatmaps for
    visualizing these counts.

    Parameters:
        project_name (str): The project name to locate the processed dataset.
        output_analysis_dir (str): Optional. The directory where the analysis results
                                   (CSV files and heatmaps) will be saved. If not provided,
                                   a default path will be used.
        heatmap_max_value (int or float): Optional. The maximum value for the heatmap color range. If provided, each heatmap will use this as the maximum count; otherwise, the maximum will be determined automatically from the data.

    Returns:
        heatmap_files (dict): A dictionary mapping each behavior (as a string) to the file path
                              of the corresponding generated heatmap (SVG file).
    """

    base_dir = os.path.join(".", "LUPEAPP_processed_dataset", project_name)
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    behaviors = load_behaviors(behaviors_file)
    if behaviors is None or not behaviors:
        raise ValueError("Failed to load behaviors or the dataset is empty.")

    frames_dir = os.path.join(base_dir, "figures", "behaviors_csv_raw-classification", "frames")
    csv_files = glob.glob(os.path.join(frames_dir, "**", "*.csv"), recursive=True)
    print("Found CSV files:", len(csv_files))
    for f in csv_files:
        print(os.path.basename(f))

    all_data_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Processing {os.path.basename(file)}: {df.shape[0]} rows")
            mouse_id = os.path.splitext(os.path.basename(file))[0]
            df['mouse_id'] = mouse_id
            df['behavior'] = df['behavior'].astype(int)
            df['behavior_label'] = df['behavior'].apply(lambda x: behavior_names[x])
            df['time_s'] = df['frame'] / 60.0

            if not df.empty:
                all_data_list.append(df)
            else:
                print(f"Warning: {file} is empty. Skipping.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data_list:
        raise ValueError("No dataframes were loaded. Please verify that your CSV files contain valid data.")

    all_data = pd.concat(all_data_list, ignore_index=True)
    print("Combined all_data shape:", all_data.shape)

    all_mice = sorted(all_data['mouse_id'].unique())
    max_time_min_bin = int(all_data['time_s'].max() // 60)
    print("Number of unique mice:", len(all_mice))
    print("Maximum minute bin:", max_time_min_bin)

    analysis_dir = output_analysis_dir or os.path.join(base_dir, "figures", "behavior_individual-mouse_screening")
    os.makedirs(analysis_dir, exist_ok=True)

    heatmap_files = {}
    for behavior in behavior_names:
        df_behavior = all_data[all_data['behavior_label'] == behavior].copy()
        df_behavior['time_min_bin'] = (df_behavior['time_s'] // 60).astype(int)

        csv_data = df_behavior.groupby(['mouse_id', 'time_min_bin']).size().unstack(fill_value=0)
        csv_data = csv_data.reindex(
            index=all_mice,
            columns=range(max_time_min_bin + 1),
            fill_value=0
        )

        csv_filename = os.path.join(analysis_dir, f"{behavior.replace(' ', '_')}_data.csv")
        csv_data.to_csv(csv_filename)
        print(f"Saved CSV data for '{behavior}' to {csv_filename}")

    # Optional: Fixed vmax for heatmaps if you want consistent coloring; otherwise, it will be determined from data.
    fixed_vmax = heatmap_max_value

    heatmap_csv_files = glob.glob(os.path.join(analysis_dir, "*_data.csv"))
    for csv_file in heatmap_csv_files:
        base_name = os.path.basename(csv_file)
        behavior_name = base_name.replace("_data.csv", "").replace("_", " ")

        data = pd.read_csv(csv_file, index_col=0)

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            data,
            cmap='Reds',
            vmin=0,
            vmax=fixed_vmax,
            cbar_kws={'label': f'{behavior_name} frames per minute'}
        )
        plt.xlabel("Time Bin (minutes)")
        plt.ylabel("Mouse ID")
        plt.title(f"{behavior_name.capitalize()} Heatmap")

        svg_filename = os.path.join(analysis_dir, f"{behavior_name.replace(' ', '_')}_heatmap.svg")
        plt.savefig(svg_filename, format='svg', bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap for '{behavior_name}' as {svg_filename}")
        heatmap_files[behavior_name] = svg_filename

    return heatmap_files
