import os
import pickle
import pandas as pd
from utils.classification import load_data, load_features
from utils.feature_utils import feature_extraction


def preprocess_data(project_name, uploaded_files, groups, conditions):
    """
    Preprocess the data uploaded via Streamlit, organized by groups and conditions.

    Parameters:
        project_name (str): The name of the project.
        uploaded_files (dict): Nested dictionary of uploaded files, grouped by group and condition.
        groups (list): List of user-defined group names.
        conditions (list): List of condition names.

    Returns:
        str: Path to the saved pickle file containing preprocessed data.
    """
    if not uploaded_files:
        raise ValueError("No files uploaded for preprocessing.")

    data = {group: {condition: {} for condition in conditions} for group in groups}
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    for i, group_name in enumerate(groups):
        group_key = f"group_{i + 1}"

        if group_key not in uploaded_files:
            raise ValueError(f"Group '{group_name}' not found in uploaded files.")

        group_files = uploaded_files[group_key]

        for condition_name in conditions:
            if condition_name not in group_files:
                raise ValueError(f"Condition '{condition_name}' not found in group '{group_name}'.")

            condition_files = group_files[condition_name]

            for file in condition_files:
                file_name = file.name
                try:
                    print(f"Processing file: {file_name}")

                    temp_file_path = os.path.join(temp_dir, file_name)
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(file.getbuffer())

                    # Read the CSV
                    data_frame = pd.read_csv(temp_file_path, encoding='utf-8', engine='python', on_bad_lines='skip')
                    data[group_name][condition_name][file_name] = data_frame

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    raise ValueError(f"Failed to process {file_name}. Ensure it's a valid CSV file.")
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

    os.rmdir(temp_dir)
    output_dir = f"../LUPEAPP_processed_dataset/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"raw_data_{project_name}.pkl")
    with open(output_file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Processed data saved to: {output_file_path}")
    return output_file_path
