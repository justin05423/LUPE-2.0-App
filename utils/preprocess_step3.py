import os
import pickle
import numpy as np
from utils.classification import load_model, load_features, load_data, weighted_smoothing
from utils.meta import *

def preprocess_get_behaviors(project_name):

    print(f"Predicting behaviors for project: {project_name}")

    # Define paths
    model_path = "./model/model.pkl"
    data_path = f"./LUPEAPP_processed_dataset/{project_name}/raw_data_{project_name}.pkl"
    features_path = f"./LUPEAPP_processed_dataset/{project_name}/binned_features_{project_name}.pkl"
    output_dir = f"./LUPEAPP_processed_dataset/{project_name}/"
    os.makedirs(output_dir, exist_ok=True)
    behaviors_file = os.path.join(output_dir, f"behaviors_{project_name}.pkl")

    # Load model, data, and features
    model = load_model(model_path)
    poses = load_data(data_path)
    features = load_features(features_path)

    # Read groups and conditions from meta.py
    try:
        groups = eval(f"groups_{project_name}")
        conditions = eval(f"conditions_{project_name}")
    except NameError as e:
        raise ValueError(f"Could not find group/condition data in meta.py: {e}")

    repeat_n = 6  # Behavior smoothing parameter

    # Initialize behaviors dictionary
    behaviors = {group: {condition: {} for condition in conditions} for group in groups}

    # Process behavior predictions
    for group in groups:
        for condition in conditions:
            if group in poses and condition in poses[group]:
                for file_name in poses[group][condition]:
                    total_n_frames = poses[group][condition][file_name].shape[0]
                    predict_ds = model.predict(features[group][condition][file_name][0])
                    predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
                    behaviors[group][condition][file_name] = weighted_smoothing(predictions, size=12)
            else:
                print(f"Warning: No data found for group '{group}' and condition '{condition}'.")

    # Save behaviors to a pickle file
    with open(behaviors_file, 'wb') as f:
        pickle.dump(behaviors, f)

    print(f"Behavior prediction complete. Data saved at: {behaviors_file}")
    return behaviors_file