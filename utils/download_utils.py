import streamlit as st
import pandas as pd
import numpy as np
from utils.feature_utils import get_avg_kinematics
from utils.feature_utils import weighted_smoothing


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def csv_predict(condition):
    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    predict_df = []
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    for f in range(len(st.session_state['features'][condition])):
        predict = st.session_state['classifier'].predict(st.session_state['features'][condition][f])
        predict_dict[f] = {'condition': np.repeat(condition, len(predict)),
                           'file': np.repeat(f+1, len(predict)),
                           'time': np.round(np.arange(0, len(predict) * 0.1, 0.1), 2),
                           'behavior': np.hstack([behavior_classes[p] for p in predict])}
        predict_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([predict_df[f] for f in range(len(predict_df))])
    return convert_df(concat_df)


def duration_pie_csv(condition, features, pose):
    if st.session_state[f'pie_table_{condition}'] is None:
        predict_dict = {key: [] for key in range(len(features))}
        duration_pie_df = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict = weighted_smoothing(predictions, size=12)
            predict_dict[f] = {'condition': np.repeat(condition, len(predict)),
                               'file': np.repeat(st.session_state[f'fnames_condition_{condition}'][f], len(predict)),
                               'time': np.round(np.arange(0, len(predict) * 0.1, 0.1), 2)[:len(predict)],
                               'behavior': predict}
            predict_df = pd.DataFrame(predict_dict[f])

            labels = predict_df['behavior'].value_counts(sort=False).index
            file_id = np.repeat(predict_df['file'].value_counts(sort=False).index,
                                len(np.unique(labels)))
            values = predict_df['behavior'].value_counts(sort=False).values
            behavior_labels = []
            for l in labels:
                behavior_labels.append(st.session_state["annotated_behaviors"][int(l)])
            # summary dataframe
            df = pd.DataFrame()
            df['condition'] = np.repeat(condition, len(values))
            df['file'] = file_id
            df['behavior'] = behavior_labels
            df['frames'] = values

            duration_pie_df.append(df)
        concat_df = pd.concat([duration_pie_df[f] for f in range(len(duration_pie_df))])
        st.session_state[f'pie_table_{condition}'] = convert_df(concat_df)
    return


def get_num_bouts(predict, behavior_classes):
    bout_counts = []
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_start_label = predict[bout_start_idx]
    for b, behavior_name in enumerate(behavior_classes):
        idx_b = np.where(bout_start_label == int(b))[0]
        if len(idx_b) > 0:
            bout_counts.append(len(idx_b))
        else:
            bout_counts.append(np.NAN)
    return bout_counts


def bout_bar_csv(condition, features, pose):
    if st.session_state[f'bar_table_{condition}'] is None:
        predict_dict = {key: [] for key in range(len(features))}
        bout_counts = {key: [] for key in range(len(features))}
        bout_counts_df = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict = weighted_smoothing(predictions, size=12)

            bout_counts[f] = get_num_bouts(predict, st.session_state["annotated_behaviors"])
            predict_dict[f] = {'condition': np.repeat(condition, len(st.session_state["annotated_behaviors"])),
                               'file': np.repeat(st.session_state[f'fnames_condition_{condition}'][f],
                                                 len(st.session_state["annotated_behaviors"])),
                               'behavior': st.session_state["annotated_behaviors"],
                               'number of bouts': bout_counts[f],
                               }
            bout_counts_df.append(pd.DataFrame(predict_dict[f]))
        concat_df = pd.concat([bout_counts_df[f] for f in range(len(bout_counts_df))])
        st.session_state[f'bar_table_{condition}'] = convert_df(concat_df)
    return


def get_duration_bouts(predict, behavior_classes, framerate=10):
    behav_durations = []
    bout_start_idx = np.where(np.diff(np.hstack([-1, predict])) != 0)[0]
    bout_durations = np.hstack([np.diff(bout_start_idx), len(predict) - np.max(bout_start_idx)])
    bout_start_label = predict[bout_start_idx]
    for b, behavior_name in enumerate(behavior_classes):
        idx_b = np.where(bout_start_label == int(b))[0]
        if len(idx_b) > 0:
            behav_durations.append(bout_durations[idx_b]/framerate)
        else:
            a = np.empty((1,))
            a[:] = np.nan
            behav_durations.append(a)
    return behav_durations


def duration_ridge_csv(condition, features, pose):
    if st.session_state[f'dur_table_{condition}'] is None:
        predict_dict = {key: [] for key in range(len(features))}
        durations_ = {key: [] for key in range(len(features))}
        durations_df = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict = weighted_smoothing(predictions, size=12)
            durations_[f] = get_duration_bouts(predict, st.session_state["annotated_behaviors"])
            predict_dict[f] = {'condition': np.hstack([np.repeat(condition, len(durations_[f][i]))
                                                       for i in range(len(durations_[f]))]),
                               'file': np.hstack([np.repeat(st.session_state[f'fnames_condition_{condition}'][f],
                                                            len(durations_[f][i]))
                                                  for i in range(len(durations_[f]))]),
                               'behavior': np.hstack([np.repeat(st.session_state["annotated_behaviors"][i],
                                                      len(durations_[f][i])) for i in range(len(durations_[f]))]),
                               'duration': np.hstack(durations_[f]),
                               }
            durations_df.append(pd.DataFrame(predict_dict[f]))
        concat_df = pd.concat([durations_df[f] for f in range(len(durations_df))])
        st.session_state[f'dur_table_{condition}'] = convert_df(concat_df)
    return


def get_transitions(predict, behavior_classes):
    class_int = [int(i) for i, behavior_name in enumerate(behavior_classes)]
    tm = [[0] * np.unique(class_int) for _ in np.unique(class_int)]
    for (i, j) in zip(predict, predict[1:]):
        tm[int(i)][int(j)] += 1
    tm_df = pd.DataFrame(tm)
    tm_array = np.array(tm)
    tm_norm = tm_array / tm_array.sum(axis=1)
    return tm_array, tm_norm


def transmat_csv(condition):
    transitions_ = []
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    for f in range(len(predict)):
        count_tm, prob_tm = get_transitions(predict[f], behavior_classes)
        transitions_.append(prob_tm)
    mean_transitions = np.mean(transitions_, axis=0)
    transmat_df = pd.DataFrame(mean_transitions, index=behavior_classes, columns=behavior_classes)
    return convert_df(transmat_df)


def kinematics_csv(condition, bp_selects):

    predict_dict = {key: [] for key in range(len(st.session_state['features'][condition]))}
    behavior_classes = st.session_state['classifier'].classes_
    names = [f'behavior {int(key)}' for key in behavior_classes]
    pose = st.session_state['pose'][condition]
    kinematics_df = []
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
        bout_disp_bps = []
        bout_duration_bps = []
        bout_avg_speed_bps = []
        for bp_select in bp_selects:
            bodypart = st.session_state['bodypart_names'].index(bp_select)
            bout_disp_all = []
            bout_duration_all = []
            bout_avg_speed_all = []
            for file_chosen in range(len(predict)):
                behavior, behavioral_start_time, behavior_duration, bout_disp, bout_duration, bout_avg_speed = \
                    get_avg_kinematics(predict[file_chosen], pose[file_chosen], bodypart, framerate=10)
                bout_disp_all.append(bout_disp)
                bout_duration_all.append(bout_duration)
                bout_avg_speed_all.append(bout_avg_speed)
            bout_disp_bps.append(bout_disp_all)
            bout_duration_bps.append(bout_duration_all)
            bout_avg_speed_bps.append(bout_avg_speed_all)
        # TODO: create dictionary with kineamtics
        # predict_dict[f] = {'condition': np.hstack([np.repeat(condition, len(durations_[f][i]))
        #                                            for i in range(len(durations_[f]))]),
        #                    'file': np.hstack([np.repeat(f, len(durations_[f][i]))
        #                                       for i in range(len(durations_[f]))]),
        #                    'behavior': np.hstack([np.repeat(behavior_classes[i],
        #                                           len(durations_[f][i])) for i in range(len(durations_[f]))]),
        #                    'distance': np.hstack(bout_disp_bps),
        #                    'duration':
        #                    'duration':
        #                    }
        kinematics_df.append(pd.DataFrame(predict_dict[f]))
    concat_df = pd.concat([kinematics_df[f] for f in range(len(kinematics_df))])
    return convert_df(concat_df)
