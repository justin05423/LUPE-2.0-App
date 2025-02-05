import os
import pickle
import streamlit as st
from utils.classification import load_data, load_features
from utils.feature_utils import feature_extraction
from utils.meta import *

def preprocess_get_features(project_name, groups=None, conditions=None, debug=False):
    """
    Extracts features from the preprocessed data and saves the binned feature file.

    Parameters:
        project_name (str): The name of the project.
        groups (list, optional): List of group names. Defaults to None.
        conditions (list, optional): List of condition names. Defaults to None.
        debug (bool, optional): If True, debug output will be printed. Defaults to False.

    Returns:
        str: Path to the saved features pickle file.
    """
    # Minimal feedback regardless of debug mode
    st.write(f"Extracting features for project: {project_name} ⌛️")

    # Define paths
    data_path = f"./LUPEAPP_processed_dataset/{project_name}/raw_data_{project_name}.pkl"
    output_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)
    features_file = os.path.join(output_dir, f"binned_features_{project_name}.pkl")

    # Load preprocessed data
    data = load_data(data_path)
    if debug:
        st.write("Loaded raw data keys:", list(data.keys()))
        for grp, cond_dict in data.items():
            st.write(f"Group: {grp}, Conditions: {list(cond_dict.keys())}")

    # If groups and conditions are None, load them from meta.py dynamically
    if groups is None or conditions is None:
        try:
            groups = eval(f"groups_{project_name}")  # e.g., ["Combined", "Male", "Female"]
            conditions = eval(f"conditions_{project_name}")  # e.g., ["EXP_YFP", "EXP_CONFON", "EXP_MORP"]
            if debug:
                st.write(f"Loaded groups from meta: {groups}")
                st.write(f"Loaded conditions from meta: {conditions}")
        except NameError as e:
            raise ValueError(f"Could not find group/condition data in meta.py: {e}")

    # Initialize feature dictionary
    features = {group: {condition: {} for condition in conditions} for group in groups}

    # Process each group and condition
    for group in groups:
        if group not in data:
            if debug:
                st.write(f"Warning: Group '{group}' not found in loaded data.")
            continue
        for condition in conditions:
            if condition not in data[group]:
                if debug:
                    st.write(f"Warning: Condition '{condition}' not found for group '{group}' in loaded data.")
                continue

            if debug:
                file_keys = list(data[group][condition].keys())
                st.write(f"Processing group '{group}', condition '{condition}', files: {file_keys}")

            for file_name, file_data in data[group][condition].items():
                if debug:
                    st.write(f"Extracting features for group '{group}', condition '{condition}', file '{file_name}'")
                    try:
                        st.write(f"Type of file_data: {type(file_data)}")
                        st.write(f"Shape of file_data: {file_data.shape}")
                    except Exception as ex:
                        st.write(f"Error getting file_data shape: {ex}")
                try:
                    # feature_extraction is expected to return a list (of length 1 in this case)
                    extracted = feature_extraction([file_data], 1, framerate=60)
                    features[group][condition][file_name] = extracted
                    if debug:
                        st.write(f"Features for '{file_name}': {extracted}")
                except Exception as e:
                    if debug:
                        st.write(f"Error extracting features for file '{file_name}': {e}")

    # Save features to a pickle file
    with open(features_file, 'wb') as f:
        pickle.dump(features, f)

    st.write(f"Feature extraction complete. Data saved at: {features_file}")
    return features_file