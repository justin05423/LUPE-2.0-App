import os
import pandas as pd
from utils.classification import load_behaviors


def behavior_csv_classification(project_name, selected_groups=None, selected_conditions=None):
    """
    Generate CSV files for behavior classifications (per frame and per second).

    Parameters:
        project_name (str): Name of the project.
        selected_groups (list, optional): List of groups to process. Process all if None.
        selected_conditions (list, optional): List of conditions to process. Process all if None.
    """
    # Define the base directory
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Load behaviors
    behaviors = load_behaviors(behaviors_file)

    # ------------------------------------------------------
    # Create CSVs for Per-Frame Behavior Classification
    # ------------------------------------------------------
    base_output_dir_frames = os.path.join(base_dir, "figures/behaviors_csv_raw-classification/frames")
    for group, conditions in behaviors.items():
        if selected_groups is not None and group not in selected_groups:
            continue
        for condition, files in conditions.items():
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            # Create a subdirectory for the group and condition
            output_dir_frames = os.path.join(base_output_dir_frames, group, condition)
            os.makedirs(output_dir_frames, exist_ok=True)
            for file_key, data in files.items():
                # Create a DataFrame: each row corresponds to a frame
                df = pd.DataFrame({
                    'frame': range(1, len(data) + 1),
                    'behavior': data
                })
                csv_filename = f'{group}_{condition}_{file_key}.csv'
                # Save DataFrame to CSV in the subfolder
                df.to_csv(os.path.join(output_dir_frames, csv_filename), index=False)
                print(f'Saved {csv_filename} in {output_dir_frames}')

    # ------------------------------------------------------
    # Create CSVs for Per-Second Behavior Classification
    # ------------------------------------------------------
    base_output_dir_seconds = os.path.join(base_dir, "figures/behaviors_csv_raw-classification/seconds")
    frame_rate = 60  # Assuming 60 frames per second
    for group, conditions in behaviors.items():
        if selected_groups is not None and group not in selected_groups:
            continue
        for condition, files in conditions.items():
            if selected_conditions is not None and condition not in selected_conditions:
                continue
            # Create a subdirectory for the group and condition
            output_dir_seconds = os.path.join(base_output_dir_seconds, group, condition)
            os.makedirs(output_dir_seconds, exist_ok=True)
            for file_key, data in files.items():
                df = pd.DataFrame({
                    'time_seconds': [i / frame_rate for i in range(len(data))],
                    'behavior': data
                })
                csv_filename = f'{group}_{condition}_{file_key}.csv'
                df.to_csv(os.path.join(output_dir_seconds, csv_filename), index=False)
                print(f'Saved {csv_filename} in {output_dir_seconds}')

    print('All files saved.')