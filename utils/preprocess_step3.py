import os
import pickle
import importlib
import numpy as np
from tqdm import tqdm
from utils.classification import load_model, load_features, load_data, weighted_smoothing, load_behaviors

def preprocess_get_behaviors(project_name):
    """
    Predict behaviors based on extracted features for each file in each group and condition.

    Parameters:
        project_name (str): The name of the project.

    Returns:
        str: Path to the saved pickle file containing predicted behaviors.
    """
    # Dynamically load group and condition data from meta.py
    try:
        meta_module = importlib.import_module("utils.meta")
        groups = getattr(meta_module, f"groups_{project_name}")
        conditions = getattr(meta_module, f"conditions_{project_name}")
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Error loading meta information for project '{project_name}': {e}")

    # Define paths
    model_path = "./model/model.pkl"
    data_path = f"./LUPEAPP_processed_dataset/{project_name}/raw_data_{project_name}.pkl"
    features_path = f"./LUPEAPP_processed_dataset/{project_name}/binned_features_{project_name}.pkl"
    output_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    output_file_path = os.path.join(output_dir, f"behaviors_{project_name}.pkl")

    # Ensure required files exist
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    # Load model, poses, and features
    print("Loading model...")
    model = load_model(model_path)

    print("Loading raw data and features...")
    poses = load_data(data_path)
    features = load_features(features_path)

    # Check if behaviors already exist
    if os.path.exists(output_file_path):
        print("Behaviors already exist. Skipping prediction.")
        return output_file_path

    # Initialize behaviors dictionary
    repeat_n = 6
    behaviors = {group: {} for group in groups}

    # Predict behaviors
    print("Generating behaviors...")
    for group in tqdm(groups, desc="Groups"):
        for condition in tqdm(conditions, desc=f"Conditions in {group}", leave=False):
            if condition not in poses[group]:
                raise ValueError(f"Condition '{condition}' not found in raw data for group '{group}'.")
            behaviors[group][condition] = {}

            for file_name in tqdm(poses[group][condition], desc=f"Processing {group}/{condition}", leave=False):
                try:
                    total_n_frames = poses[group][condition][file_name].shape[0]
                    # Predict downsampled features
                    predict_ds = model.predict(features[group][condition][file_name][0])
                    # Upsample predictions by duplicating
                    predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
                    # Smooth predictions
                    behaviors[group][condition][file_name] = weighted_smoothing(predictions, size=12)
                except Exception as e:
                    raise ValueError(f"Error predicting behaviors for {file_name}: {e}")

    # Save behaviors to a pickle file
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, 'wb') as f:
        pickle.dump(behaviors, f)

    print(f"Predicted behaviors saved to: {output_file_path}")
    return output_file_path