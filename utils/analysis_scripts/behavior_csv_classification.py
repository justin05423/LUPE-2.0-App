# utils/analysis_scripts/behavior_csv_classification.py

import os
import pandas as pd
from utils.classification import load_behaviors


def behavior_csv_classification(project_name):
    """
    Generate CSV files for behavior classifications (per frame and per second).

    Parameters:
        project_name (str): Name of the project.
    """
    # Define the base directory
    base_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    behaviors_file = os.path.join(base_dir, f"behaviors_{project_name}.pkl")

    # Load behaviors
    behaviors = load_behaviors(behaviors_file)

    # Directory to save the CSV files (per frame)
    output_dir_frames = os.path.join(base_dir, "figures/behaviors_csv_raw-classification/frames")
    os.makedirs(output_dir_frames, exist_ok=True)

    # Behaviors per frame
    for group, conditions in behaviors.items():
        for condition, files in conditions.items():
            for file_key, data in files.items():
                # Create a DataFrame from the data
                df = pd.DataFrame({'frame': range(1, len(data) + 1), 'behavior': data})

                # Construct the filename
                csv_filename = f'{group}_{condition}_{file_key}.csv'

                # Save the DataFrame to a CSV file
                df.to_csv(os.path.join(output_dir_frames, csv_filename), index=False)
                print(f'Saved {csv_filename}')

    # Directory to save the CSV files (per second)
    output_dir_seconds = os.path.join(base_dir, "figures/behaviors_csv_raw-classification/seconds")
    os.makedirs(output_dir_seconds, exist_ok=True)

    # Behaviors per second
    frame_rate = 60  # Assuming 60 frames per second
    for group, conditions in behaviors.items():
        for condition, files in conditions.items():
            for file_key, data in files.items():
                # Create a DataFrame from the data
                df = pd.DataFrame({'time_seconds': [i / frame_rate for i in range(len(data))], 'behavior': data})

                # Construct the filename
                csv_filename = f'{group}_{condition}_{file_key}.csv'

                # Save the DataFrame to a CSV file
                df.to_csv(os.path.join(output_dir_seconds, csv_filename), index=False)
                print(f'Saved {csv_filename}')

    print('All files saved.')