import pickle
import pandas as pd
from utils.feature_utils import feature_extraction, weighted_smoothing
import numpy as np
import joblib
import os


def load_model(path):
    with open(path, 'rb') as fr:
        model = pickle.load(fr)
    return model


def load_data(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    return data


def load_features(path):
    with open(path, 'rb') as fr:
        features = pickle.load(fr)
    return features


def load_model_features(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'feats_targets.sav')
    # config, _ = load_config(os.path.join(path, name))

    return [i for i in data]


def load_embeddings(path, name):
    data = _load_sav(path, name, 'embedding_output.sav')
    # config, _ = load_config(os.path.join(path, name))

    return [i for i in data]


def load_behaviors(path):
    with open(path, 'rb') as fr:
        behaviors = pickle.load(fr)
    return behaviors


def pie_predict(condition, model, poses, behavior_names, repeat_n=6):
    features = feature_extraction(poses, len(poses), framerate=60)
    predict = []
    for f, feat in enumerate(features):
        total_n_frames = poses[f].shape[0]
        predict_ds = model.predict(feat)
        predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
        predict.append(weighted_smoothing(predictions, size=12))

    predict_dict = {'condition': np.repeat(condition, len(np.hstack(predict))),
                    'behavior': np.hstack(predict)}
    df_raw = pd.DataFrame(data=predict_dict)
    labels = df_raw['behavior'].value_counts(sort=False).index
    values = df_raw['behavior'].value_counts(sort=False).values
    # summary dataframe
    df = pd.DataFrame()
    behavior_labels = []
    for l in labels:
        behavior_labels.append(behavior_names[int(l)])
    df["values"] = values
    df['labels'] = behavior_labels
    return df


def _load_sav(path, name, filename):
    """just a simplification for all those load functions"""
    with open(os.path.join(path, name, filename), 'rb') as fr:
        data = joblib.load(fr)
    return data


def load_all_train(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'all_train.sav')
    return [i for i in data]


def load_iter0(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iter0.sav')
    return [i for i in data]


def load_iterX(path, name):
    # working dir is already the prefix (if user directly put in the project folder as working dir)
    data = _load_sav(path, name, 'iterX.sav')
    return [i for i in data]
