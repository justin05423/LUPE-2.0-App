import stat

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import stqdm
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter
import warnings

from utils.classifier_utils import *
from utils.download_utils import *


def ethogram_plot(condition, new_predictions, behavior_names, behavior_colors, length_):
    colL, colR = st.columns(2)
    if len(new_predictions) == 1:
        colL.markdown(':orange[1] file only')
        f_select = 0
    else:
        f_select = colL.slider('select file to generate ethogram',
                               min_value=1, max_value=len(new_predictions), value=1,
                               key=f'ethogram_slider_{condition}')
    file_idx = f_select - 1
    prefill_array = np.zeros((len(new_predictions[file_idx]),
                              len(st.session_state['classifier'].classes_)))
    default_colors_wht = ['w']
    default_colors_wht.extend(behavior_colors)
    cmap_ = ListedColormap(default_colors_wht)

    count = 0
    for b in np.unique(st.session_state['classifier'].classes_):
        idx_b = np.where(new_predictions[file_idx] == b)[0]
        prefill_array[idx_b, count] = b + 1
        count += 1
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    seed_num = colR.number_input('seed for segment',
                                 min_value=0, max_value=None, value=42,
                                 key=f'cond{condition}_seed')
    np.random.seed(seed_num)
    behaviors_with_names = behavior_names
    if colL.checkbox('use randomized time',
                     value=True,
                     key=f'cond{condition}_ckbx'):
        rand_start = np.random.choice(prefill_array.shape[0] - length_, 1, replace=False)
        ax.imshow(prefill_array[int(rand_start):int(rand_start + length_), :].T, cmap=cmap_)
        ax.set_xticks(np.arange(0, length_, int(length_ / 5)))
        ax.set_xticklabels(np.arange(int(rand_start), int(rand_start + length_), int(length_ / 5)) / 10)
        ax.set_yticks(np.arange(len(behaviors_with_names)))
        ax.set_yticklabels(behaviors_with_names)
        ax.set_xlabel('seconds')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        rand_start = 0
        ax.imshow(prefill_array[rand_start:rand_start + length_, :].T, cmap=cmap_)
        ax.set_xticks(np.arange(rand_start, length_, int(length_ / 5)))
        ax.set_xticklabels(np.arange(0, length_, int(length_ / 5)) / 10)
        ax.set_yticks(np.arange(len(behaviors_with_names)))
        ax.set_yticklabels(behaviors_with_names)
        ax.set_xlabel('seconds')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, prefill_array, rand_start


def ethogram_predict(placeholder, condition, behavior_colors, length_):
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))

    with placeholder:
        etho_placeholder = st.empty()
        fig, prefill_array, rand_start = ethogram_plot(condition, predict, behavior_classes,
                                                       list(behavior_colors.values()), length_)
        etho_placeholder.pyplot(fig)


def condition_etho_plot():
    behavior_classes = st.session_state['classifier'].classes_
    length_container = st.container()
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    behavior_colors = {key: [] for key in behavior_classes}
    all_c_options = list(mcolors.CSS4_COLORS.keys())
    np.random.seed(42)
    selected_idx = np.random.choice(np.arange(len(all_c_options)),
                                    len(behavior_classes),
                                    replace=False)
    default_colors = [all_c_options[s] for s in selected_idx]
    col1, col2, col3, col4 = option_expander.columns(4)
    for i, class_id in enumerate(behavior_classes):
        if i % 4 == 0:
            behavior_colors[class_id] = col1.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )

        elif i % 4 == 1:
            behavior_colors[class_id] = col2.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 2:
            behavior_colors[class_id] = col3.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
        elif i % 4 == 3:
            behavior_colors[class_id] = col4.selectbox(f'select color for {behavior_classes[i]}',
                                                       all_c_options,
                                                       index=all_c_options.index(default_colors[i]),
                                                       key=f'color_option{i}'
                                                       )
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    length_ = length_container.slider('number of frames',
                                      min_value=25, max_value=250,
                                      value=75,
                                      key=f'length_slider')
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        ethogram_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         behavior_colors, length_)
        predict_csv = csv_predict(
            list(st.session_state['features'].keys())[count],
        )

        left_expander.download_button(
            label="Download data as CSV",
            data=predict_csv,
            file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                ethogram_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 behavior_colors, length_)
                predict_csv = csv_predict(
                    list(st.session_state['features'].keys())[count],
                )

                right_expander.download_button(
                    label="Download data as CSV",
                    data=predict_csv,
                    file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            ethogram_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             behavior_colors, length_)
            predict_csv = csv_predict(
                list(st.session_state['features'].keys())[count],
            )

            right_expander.download_button(
                label="Download data as CSV",
                data=predict_csv,
                file_name=f"predictions_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def pie_predict(placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'pie_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        # TODO: find a color workaround if a class is missing
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
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
            behavior_labels.append(st.session_state['annotated_behaviors'][int(l)])
        df["values"] = values
        df['labels'] = behavior_labels
        df["colors"] = df["labels"].apply(lambda x:
                                          behavior_colors.get(x))  # to connect Column value to Color in Dict
        st.session_state[f'pie_{condition}'] = df

    with placeholder:
        fig = go.Figure(data=[go.Pie(labels=st.session_state[f'pie_{condition}']["labels"],
                                     values=st.session_state[f'pie_{condition}']["values"],
                                     hole=.4)])
        fig.update_traces(hoverinfo='label+percent',
                          textinfo='value',
                          textfont_size=16,
                          marker=dict(colors=st.session_state[f'pie_{condition}']["colors"],
                                      line=dict(color='#000000', width=1)))
        st.plotly_chart(fig, use_container_width=True)


def condition_pie_plot():
    figure_container = st.container()
    rows = int(np.ceil(st.session_state['num_condition'] / 2))
    mod_ = st.session_state['num_condition'] % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                          f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 1}</h1> "
                          , unsafe_allow_html=True)
        if f'pie_{count + 1}' not in st.session_state:
            st.session_state[f'pie_{count + 1}'] = None
        if f'pie_table_{count + 1}' not in st.session_state:
            st.session_state[f'pie_table_{count + 1}'] = None
        pie_predict(left_col,
                    count + 1,
                    st.session_state[f'feats_condition_{count + 1}'],
                    st.session_state[f'filt_pose_condition_{count + 1}'],
                    st.session_state['behavior_colors'])

        duration_pie_csv(
            count + 1,
            st.session_state[f'feats_condition_{count + 1}'],
            st.session_state[f'filt_pose_condition_{count + 1}']
        )
        left_col.download_button(
            label="Download data as CSV",
            data=st.session_state[f'pie_table_{count + 1}'],
            file_name=f"total_durations_condition_{count + 1}.csv",
            mime='text/csv',
            key=f"{count + 1}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                                   f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                                   , unsafe_allow_html=True)
                if f'pie_{count + 1}' not in st.session_state:
                    st.session_state[f'pie_{count + 1}'] = None
                if f'pie_table_{count + 1}' not in st.session_state:
                    st.session_state[f'pie_table_{count + 1}'] = None
                pie_predict(right_col,
                            count + 1,
                            st.session_state[f'feats_condition_{count + 1}'],
                            st.session_state[f'filt_pose_condition_{count + 1}'],
                            st.session_state['behavior_colors'])
                duration_pie_csv(
                    count + 1,
                    st.session_state[f'feats_condition_{count + 1}'],
                    st.session_state[f'filt_pose_condition_{count + 1}']
                )
                right_col.download_button(
                    label="Download data as CSV",
                    data=st.session_state[f'pie_table_{count + 1}'],
                    file_name=f"total_durations_condition_{count + 1}.csv",
                    mime='text/csv',
                    key=f"{count + 1}_dwnload"
                )
                count += 1
        else:
            right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                               f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                               , unsafe_allow_html=True)
            if f'pie_{count + 1}' not in st.session_state:
                st.session_state[f'pie_{count + 1}'] = None
            if f'pie_table_{count + 1}' not in st.session_state:
                st.session_state[f'pie_table_{count + 1}'] = None
            pie_predict(right_col,
                        count + 1,
                        st.session_state[f'feats_condition_{count + 1}'],
                        st.session_state[f'filt_pose_condition_{count + 1}'],
                        st.session_state['behavior_colors'])
            duration_pie_csv(
                count + 1,
                st.session_state[f'feats_condition_{count + 1}'],
                st.session_state[f'filt_pose_condition_{count + 1}']
            )
            right_col.download_button(
                label="Download data as CSV",
                data=st.session_state[f'pie_table_{count + 1}'],
                file_name=f"total_durations_condition_{count + 1}.csv",
                mime='text/csv',
                key=f"{count + 1}_dwnload"
            )
            count += 1


def location_predict(behav_selects, placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'location_pred_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict.append(weighted_smoothing(predictions, size=12))
        st.session_state[f'location_pred_{condition}'] = predict
    # tail-base as position indicator
    bodypart_idx = st.session_state['bodypart_names'].index('tail_base') * 2
    if len(st.session_state[f'location_pred_{condition}']) == 1:
        placeholder.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                             f"font-family:Avenir; font-weight:normal'>1 file only</h1> "
                             , unsafe_allow_html=True)
        file_chosen = 0
    else:
        f_select = placeholder.select_slider('Select file',
                                             options=st.session_state[f'fnames_condition_{condition}'],
                                             value=st.session_state[f'fnames_condition_{condition}'][0],
                                             key=f'location_slider_{condition}')
        file_chosen = st.session_state[f'fnames_condition_{condition}'].index(f_select)

    fig = plt.figure(facecolor='#0C0C0C')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(None)

    center = (768 / 2, 770 / 2)
    radius = 768 / 2 + 20
    h = '00FEFF'  # cyan like the logo
    rgb_val = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    circle = Circle(center, radius, color=rgb_val, linewidth=3, fill=False)

    ax.add_patch(circle)
    ax.set_aspect('equal')
    for b in [st.session_state["annotated_behaviors"].index(behav_selects[i])
              for i in range(len(behav_selects))]:
        idx_b = np.where(st.session_state[f'location_pred_{condition}'][file_chosen] == b)[0]
        ax.scatter(pose[file_chosen][idx_b, bodypart_idx],
                   pose[file_chosen][idx_b, bodypart_idx + 1],
                   c=behavior_colors[st.session_state["annotated_behaviors"][b]],
                   s=0.7)
    ax.invert_yaxis()
    plt.axis('off')
    plt.axis('equal')
    placeholder.pyplot(fig)


def location_heatmap(placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'location_pred_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict.append(weighted_smoothing(predictions, size=12))
        st.session_state[f'location_pred_{condition}'] = predict
    placeholder.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                         f"font-family:Avenir; font-weight:normal'>Select behavior</h1> "
                         , unsafe_allow_html=True)
    behav_selects = placeholder.multiselect('select behavior',
                                            st.session_state["annotated_behaviors"],
                                            default='walking',
                                            key=f'behavior_multiselect_{condition}_2',
                                            label_visibility='collapsed')

    # tail-base as position indicator
    bodypart_idx = st.session_state['bodypart_names'].index('tail_base') * 2

    fig = plt.figure(facecolor='#0C0C0C')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(None)

    center = (768 / 2, 770 / 2)
    radius = 768 / 2 + 20
    h = '00FEFF'  # cyan like the logo
    rgb_val = tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    circle = Circle(center, radius, color=rgb_val, linewidth=3, fill=False)
    beh_colors = []
    for b in [st.session_state["annotated_behaviors"].index(behav_selects[i])
              for i in range(len(behav_selects))]:
        hist2d_all = []
        colors = ['#0C0C0C', behavior_colors[st.session_state["annotated_behaviors"][b]]]
        beh_colors.append(behavior_colors[st.session_state["annotated_behaviors"][b]])
        cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
        for file_chosen in range(len(st.session_state[f'fnames_condition_{condition}'])):
            idx_b = np.where(st.session_state[f'location_pred_{condition}'][file_chosen] == b)[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                heatmap, xedges, yedges = np.histogram2d(pose[file_chosen][idx_b, bodypart_idx],
                                                         pose[file_chosen][idx_b, bodypart_idx + 1],
                                                         bins=[np.arange(0, 768, 20), np.arange(0, 770, 20)],
                                                         density=True)
            heatmap[heatmap == 0] = np.nan
            hist2d_all.append(heatmap)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ax.imshow(np.nanmean(hist2d_all, axis=0).T,
                      extent=extent, origin='lower', cmap=cm)
    patches = [mpatches.Patch(color=beh_colors[i], label=behav_selects[i]) for i in range(len(behav_selects))]
    lgd = ax.legend(handles=patches, facecolor="#0C0C0C", frameon=False, prop={"size": 16},
                    ncol=2, bbox_to_anchor=(0.5, -0.5), loc='lower center', edgecolor='w')
    for text in lgd.get_texts():
        text.set_color("#EEEEEE")
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.axis('off')
    plt.axis('equal')
    placeholder.pyplot(fig)
    return behav_selects


def condition_location_plot():
    figure_container = st.container()
    rows = int(np.ceil(st.session_state['num_condition'] / 2))
    mod_ = st.session_state['num_condition'] % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                          f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 1}</h1> "
                          , unsafe_allow_html=True)
        if f'location_pred_{count + 1}' not in st.session_state:
            st.session_state[f'location_pred_{count + 1}'] = None

        behav_selects = location_heatmap(left_col,
                                         count + 1,
                                         st.session_state[f'feats_condition_{count + 1}'],
                                         st.session_state[f'filt_pose_condition_{count + 1}'],
                                         st.session_state['behavior_colors'])
        example_location = left_col.checkbox('show examples?', key=f'examp_ckbx{count + 1}')
        if example_location:
            location_predict(behav_selects,
                             left_col,
                             count + 1,
                             st.session_state[f'feats_condition_{count + 1}'],
                             st.session_state[f'filt_pose_condition_{count + 1}'],
                             st.session_state['behavior_colors'])
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                                   f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                                   , unsafe_allow_html=True)
                if f'location_pred_{count + 1}' not in st.session_state:
                    st.session_state[f'location_pred_{count + 1}'] = None

                behav_selects = location_heatmap(right_col,
                                                 count + 1,
                                                 st.session_state[f'feats_condition_{count + 1}'],
                                                 st.session_state[f'filt_pose_condition_{count + 1}'],
                                                 st.session_state['behavior_colors'])
                example_location = right_col.checkbox('show examples?', key=f'examp_ckbx{count + 1}')
                if example_location:
                    location_predict(behav_selects,
                                     right_col,
                                     count + 1,
                                     st.session_state[f'feats_condition_{count + 1}'],
                                     st.session_state[f'filt_pose_condition_{count + 1}'],
                                     st.session_state['behavior_colors'])
                count += 1
        else:
            right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                               f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                               , unsafe_allow_html=True)
            if f'location_pred_{count + 1}' not in st.session_state:
                st.session_state[f'location_pred_{count + 1}'] = None
            behav_selects = location_heatmap(right_col,
                                             count + 1,
                                             st.session_state[f'feats_condition_{count + 1}'],
                                             st.session_state[f'filt_pose_condition_{count + 1}'],
                                             st.session_state['behavior_colors'])
            example_location = right_col.checkbox('show examples?', key=f'examp_ckbx{count + 1}')
            if example_location:
                location_predict(behav_selects,
                                 right_col,
                                 count + 1,
                                 st.session_state[f'feats_condition_{count + 1}'],
                                 st.session_state[f'filt_pose_condition_{count + 1}'],
                                 st.session_state['behavior_colors'])
            count += 1


def bar_predict(placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'bar_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict.append(weighted_smoothing(predictions, size=12))

        bout_counts = []
        for file_idx in range(len(predict)):
            bout_counts.append(get_num_bouts(predict[file_idx], st.session_state["annotated_behaviors"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bout_mean = np.nanmean(bout_counts, axis=0)
            bout_std = np.nanstd(bout_counts, axis=0)
        st.session_state[f'bar_{condition}'] = [bout_mean, bout_std]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=st.session_state["annotated_behaviors"],
        x=st.session_state[f'bar_{condition}'][0],
        # name=names,
        error_x=dict(type='data', array=st.session_state[f'bar_{condition}'][1]),
        width=0.5,
        marker_color=pd.Series(behavior_colors, dtype='object'),
        marker_line=dict(width=0.3, color='whitesmoke'))
    )
    fig.update_traces(orientation='h')
    fig.update_layout(xaxis=dict(title=f"counts (mean+-sd) across "
                                       f"{len(st.session_state[f'feats_condition_{condition}'])} files"),
                      )
    fig['layout']['yaxis']['autorange'] = "reversed"
    with placeholder:
        st.plotly_chart(fig, use_container_width=True)


def condition_bar_plot():
    figure_container = st.container()
    rows = int(np.ceil(st.session_state['num_condition'] / 2))
    mod_ = st.session_state['num_condition'] % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                          f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 1}</h1> "
                          , unsafe_allow_html=True)
        if f'bar_{count + 1}' not in st.session_state:
            st.session_state[f'bar_{count + 1}'] = None
        if f'bar_table_{count + 1}' not in st.session_state:
            st.session_state[f'bar_table_{count + 1}'] = None
        bar_predict(left_col,
                    count + 1,
                    st.session_state[f'feats_condition_{count + 1}'],
                    st.session_state[f'filt_pose_condition_{count + 1}'],
                    st.session_state['behavior_colors'])
        bout_bar_csv(
            count + 1,
            st.session_state[f'feats_condition_{count + 1}'],
            st.session_state[f'filt_pose_condition_{count + 1}']
        )
        left_col.download_button(
            label="Download data as CSV",
            data=st.session_state[f'bar_table_{count + 1}'],
            file_name=f"bouts_condition_{count + 1}.csv",
            mime='text/csv',
            key=f"{count + 1}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                                   f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                                   , unsafe_allow_html=True)
                if f'bar_{count + 1}' not in st.session_state:
                    st.session_state[f'bar_{count + 1}'] = None
                if f'bar_table_{count + 1}' not in st.session_state:
                    st.session_state[f'bar_table_{count + 1}'] = None
                bar_predict(right_col,
                            count + 1,
                            st.session_state[f'feats_condition_{count + 1}'],
                            st.session_state[f'filt_pose_condition_{count + 1}'],
                            st.session_state['behavior_colors'])
                bout_bar_csv(
                    count + 1,
                    st.session_state[f'feats_condition_{count + 1}'],
                    st.session_state[f'filt_pose_condition_{count + 1}']
                )
                right_col.download_button(
                    label="Download data as CSV",
                    data=st.session_state[f'bar_table_{count + 1}'],
                    file_name=f"bouts_condition_{count + 1}.csv",
                    mime='text/csv',
                    key=f"{count + 1}_dwnload"
                )
                count += 1
        else:
            right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                               f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                               , unsafe_allow_html=True)
            if f'bar_{count + 1}' not in st.session_state:
                st.session_state[f'bar_{count + 1}'] = None
            if f'bar_table_{count + 1}' not in st.session_state:
                st.session_state[f'bar_table_{count + 1}'] = None
            bar_predict(right_col,
                        count + 1,
                        st.session_state[f'feats_condition_{count + 1}'],
                        st.session_state[f'filt_pose_condition_{count + 1}'],
                        st.session_state['behavior_colors'])
            bout_bar_csv(
                count + 1,
                st.session_state[f'feats_condition_{count + 1}'],
                st.session_state[f'filt_pose_condition_{count + 1}']
            )
            right_col.download_button(
                label="Download data as CSV",
                data=st.session_state[f'bar_table_{count + 1}'],
                file_name=f"bouts_condition_{count + 1}.csv",
                mime='text/csv',
                key=f"{count + 1}_dwnload"
            )
            count += 1


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def ridge_predict(placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'dur_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        # TODO: find a color workaround if a class is missing
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict.append(weighted_smoothing(predictions, size=12))
        duration_ = []
        for file_idx in range(len(predict)):
            duration_.append(get_duration_bouts(predict[file_idx],
                                                st.session_state["annotated_behaviors"],
                                                framerate=60))
        for file_chosen in range(len(duration_)):
            if file_chosen == 0:
                duration_matrix = boolean_indexing(duration_[file_chosen])
            else:
                duration_matrix = np.hstack((duration_matrix, boolean_indexing(duration_[file_chosen])))
        st.session_state[f'dur_{condition}'] = duration_matrix
    colors = [mcolors.to_hex(i) for i in list(behavior_colors.values())]
    fig = go.Figure()
    for data_line, color, name in zip(st.session_state[f'dur_{condition}'], colors,
                                      st.session_state["annotated_behaviors"]):
        fig.add_trace(go.Box(x=data_line,
                             jitter=0.5,
                             whiskerwidth=0.3,
                             fillcolor=color,
                             marker_size=2,
                             line_width=1.2,
                             line_color='#EEEEEE',
                             name=name))
    fig.update_traces(orientation='h')
    fig.update_layout(xaxis=dict(title='bout duration (seconds)'),
                      )
    fig['layout']['yaxis']['autorange'] = "reversed"
    with placeholder:
        st.plotly_chart(fig, use_container_width=True)


def condition_ridge_plot():
    figure_container = st.container()
    rows = int(np.ceil(st.session_state['num_condition'] / 2))
    mod_ = st.session_state['num_condition'] % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                          f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 1}</h1> "
                          , unsafe_allow_html=True)
        if f'dur_{count + 1}' not in st.session_state:
            st.session_state[f'dur_{count + 1}'] = None
        if f'dur_table_{count + 1}' not in st.session_state:
            st.session_state[f'dur_table_{count + 1}'] = None
        ridge_predict(left_col,
                      count + 1,
                      st.session_state[f'feats_condition_{count + 1}'],
                      st.session_state[f'filt_pose_condition_{count + 1}'],
                      st.session_state['behavior_colors'])
        duration_ridge_csv(
            count + 1,
            st.session_state[f'feats_condition_{count + 1}'],
            st.session_state[f'filt_pose_condition_{count + 1}']
        )
        left_col.download_button(
            label="Download data as CSV",
            data=st.session_state[f'dur_table_{count + 1}'],
            file_name=f"bout_durations_condition_{count + 1}.csv",
            mime='text/csv',
            key=f"{count + 1}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                                   f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                                   , unsafe_allow_html=True)
                if f'dur_{count + 1}' not in st.session_state:
                    st.session_state[f'dur_{count + 1}'] = None
                if f'dur_table_{count + 1}' not in st.session_state:
                    st.session_state[f'dur_table_{count + 1}'] = None
                ridge_predict(right_col,
                              count + 1,
                              st.session_state[f'feats_condition_{count + 1}'],
                              st.session_state[f'filt_pose_condition_{count + 1}'],
                              st.session_state['behavior_colors'])
                duration_ridge_csv(
                    count + 1,
                    st.session_state[f'feats_condition_{count + 1}'],
                    st.session_state[f'filt_pose_condition_{count + 1}']
                )
                right_col.download_button(
                    label="Download data as CSV",
                    data=st.session_state[f'dur_table_{count + 1}'],
                    file_name=f"bout_durations_condition_{count + 1}.csv",
                    mime='text/csv',
                    key=f"{count + 1}_dwnload"
                )
                count += 1
        else:
            right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                               f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                               , unsafe_allow_html=True)
            if f'dur_{count + 1}' not in st.session_state:
                st.session_state[f'dur_{count + 1}'] = None
            if f'dur_table_{count + 1}' not in st.session_state:
                st.session_state[f'dur_table_{count + 1}'] = None
            ridge_predict(right_col,
                          count + 1,
                          st.session_state[f'feats_condition_{count + 1}'],
                          st.session_state[f'filt_pose_condition_{count + 1}'],
                          st.session_state['behavior_colors'])
            duration_ridge_csv(
                count + 1,
                st.session_state[f'feats_condition_{count + 1}'],
                st.session_state[f'filt_pose_condition_{count + 1}']
            )
            right_col.download_button(
                label="Download data as CSV",
                data=st.session_state[f'dur_table_{count + 1}'],
                file_name=f"bout_durations_condition_{count + 1}.csv",
                mime='text/csv',
                key=f"{count + 1}_dwnload"
            )
            count += 1


def transmat_predict(placeholder, condition, heatmap_color_scheme):
    # behavior_classes = st.session_state['classifier'].classes_
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        transitions_ = []
        for file_idx in range(len(predict)):
            count_tm, prob_tm = get_transitions(predict[file_idx], behavior_classes)
            transitions_.append(prob_tm)
        mean_transitions = np.mean(transitions_, axis=0)
        fig = px.imshow(mean_transitions,
                        color_continuous_scale=heatmap_color_scheme,
                        aspect='equal'
                        )
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(behavior_classes)),
                ticktext=behavior_classes),
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(behavior_classes)),
                ticktext=behavior_classes)
        )
        st.plotly_chart(fig, use_container_width=True)


def directedgraph_predict(placeholder, condition, heatmap_color_scheme):
    # behavior_classes = st.session_state['classifier'].classes_
    # names = [f'behavior {int(key)}' for key in behavior_classes]
    behavior_classes = [st.session_state['annotations'][i]['name']
                        for i in list(st.session_state['annotations'])]
    predict = []
    for f in range(len(st.session_state['features'][condition])):
        predict.append(st.session_state['classifier'].predict(st.session_state['features'][condition][f]))
    with placeholder:
        transitions_count = []
        transitions_prob = []
        for file_idx in range(len(predict)):
            count_tm, prob_tm = get_transitions(predict[file_idx], behavior_classes)
            transitions_count.append(count_tm)
            transitions_prob.append(prob_tm)
        transition_count_mean = np.nanmean(transitions_count, axis=0)
        transitions_prob_mean = np.nanmean(transitions_prob, axis=0)
        diag = [transition_count_mean[i][i] for i in range(len(transition_count_mean))]
        diag_p = np.array(diag) / np.array(diag).max()
        # keep diag to provide information about relative behavioral duration
        # scale it by 50, works well, and save it as a global variable
        node_sizes = [50 * i for i in diag_p]
        ## transition matrix from 2d array into numpy matrix for networkx
        transition_prob_raw = np.matrix(transitions_prob_mean)
        # replace diagonal with 0
        np.fill_diagonal(transition_prob_raw, 0)
        transition_prob_norm = transition_prob_raw / transition_prob_raw.sum(axis=1)
        nan_indices = np.isnan(transition_prob_norm)
        transition_prob_norm[nan_indices] = 0

        fig = plt.figure(figsize=(8, 8))
        # particular networkx graph
        graph = nx.from_numpy_array(transition_prob_norm, create_using=nx.MultiDiGraph())
        # set node position with seed 0 for reproducibility
        node_position = nx.layout.spring_layout(graph, seed=0)
        # edge colors is equivalent to the weight
        edge_colors = [graph[u][v][0].get('weight') for u, v in graph.edges()]
        # TODO: try to find a way to fix vmin vmax in directed graph
        # c_max = np.max(edge_colors)
        # max_c = st.slider('color axis limit',
        #                   min_value=0.0,
        #                   max_value=1.0,
        #                   value=np.float(c_max),
        #                   key=f'max_color_slider_{condition}')

        # node is dependent on the self transitions, which is defined in compute dynamics above, use blue colormap
        nodes = nx.draw_networkx_nodes(graph, node_position, node_size=node_sizes,
                                       node_color='blue')

        # edges are drawn as arrows with blue colormap, size 8 with width 1.5
        edges = nx.draw_networkx_edges(graph, node_position, node_size=node_sizes, arrowstyle='->',
                                       arrowsize=8, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=1.5)
        # label position is 0.005 to the right of the node
        label_pos = [node_position[i] + 0.005 for i in range(len(node_position))]
        # draw labels with font size 10
        labels_dict = {}
        for i, label in enumerate(behavior_classes):
            labels_dict[i] = label
        nx.draw_networkx_labels(graph, label_pos, labels_dict, font_size=10)
        # generate colorbar from the edge colors
        pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
        # pc.set_clim([0, max_c])
        pc.set_array(edge_colors)
        plt.colorbar(pc, shrink=0.5, location='bottom')
        ax = plt.gca()
        ax.set_axis_off()
        st.pyplot(fig, use_container_width=True)


def condition_transmat_plot():
    figure_container = st.container()
    option_expander = st.expander("Configure colors",
                                  expanded=True)
    named_colorscales = px.colors.named_colorscales()
    col1, col2 = option_expander.columns([3, 1])
    heatmap_color_scheme = col1.selectbox(f'select colormap for heatmap',
                                          named_colorscales,
                                          index=named_colorscales.index('agsunset'),
                                          key='color_scheme')
    col2.write('')
    col2.write('')
    if col2.checkbox('reverse?'):
        heatmap_color_scheme = str.join('', (heatmap_color_scheme, '_r'))
    num_cond = len(st.session_state['features'])
    rows = int(np.ceil(num_cond / 2))
    mod_ = num_cond % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_expander = left_col.expander(f'Condition {row * 2 + 1}:',
                                          expanded=True)
        transmat_predict(left_expander,
                         list(st.session_state['features'].keys())[count],
                         heatmap_color_scheme)
        directedgraph_predict(left_expander,
                              list(st.session_state['features'].keys())[count],
                              heatmap_color_scheme)
        transition_csv = transmat_csv(
            list(st.session_state['features'].keys())[count],
        )
        left_expander.download_button(
            label="Download data as CSV",
            data=transition_csv,
            file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
            mime='text/csv',
            key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                    expanded=True)
                transmat_predict(right_expander,
                                 list(st.session_state['features'].keys())[count],
                                 heatmap_color_scheme)
                directedgraph_predict(right_expander,
                                      list(st.session_state['features'].keys())[count],
                                      heatmap_color_scheme)
                transition_csv = transmat_csv(
                    list(st.session_state['features'].keys())[count],
                )
                right_expander.download_button(
                    label="Download data as CSV",
                    data=transition_csv,
                    file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
                    mime='text/csv',
                    key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                )
                count += 1
        else:
            right_expander = right_col.expander(f'Condition {row * 2 + 2}:',
                                                expanded=True)
            transmat_predict(right_expander,
                             list(st.session_state['features'].keys())[count],
                             heatmap_color_scheme)
            directedgraph_predict(right_expander,
                                  list(st.session_state['features'].keys())[count],
                                  heatmap_color_scheme)
            transition_csv = transmat_csv(
                list(st.session_state['features'].keys())[count],
            )
            right_expander.download_button(
                label="Download data as CSV",
                data=transition_csv,
                file_name=f"transitions_{list(st.session_state['features'].keys())[count]}.csv",
                mime='text/csv',
                key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            )
            count += 1


def kinematix_predict(placeholder, condition, features, pose, behavior_colors):
    if st.session_state[f'kine_pred_{condition}'] is None:
        predict = []
        repeat_n = int(60 / 10)
        for f, feat in enumerate(features):
            total_n_frames = pose[f].shape[0]
            predict_ds = st.session_state['classifier'].predict(feat)
            predictions = np.pad(predict_ds.repeat(repeat_n), (repeat_n, 0), 'edge')[:total_n_frames]
            predict.append(weighted_smoothing(predictions, size=12))
        st.session_state[f'speed_pred_{condition}'] = predict
    with placeholder:
        bp_selects = st.selectbox('select body part',
                                  st.session_state['bodypart_names'],
                                  index=st.session_state['bodypart_names'].index("r_hindpaw"),
                                  key=f'bodypart_selectbox_{condition}')
        bout_disp_bps = []
        # bout_duration_bps = []
        # bout_avg_speed_bps = []
        # behavior_all_bps = []
        for bp_select in [bp_selects]:
            bodypart = st.session_state['bodypart_names'].index(bp_select)
            bout_disp_all = []
            # behavior_all = []
            for file_chosen in range(len(predict)):
                behavior, behavioral_start_time, behavior_duration, bout_disp, bout_duration, bout_avg_speed = \
                    get_avg_kinematics(predict[file_chosen], pose[file_chosen], bodypart, framerate=60)
                bout_disp_all.append(bout_disp)
                # bout_duration_all.append(bout_duration)
                # bout_avg_speed_all.append(bout_avg_speed)
                # behavior_all.append(behavior)
            # per bout, average total bodypart movement in pixel displacement,
            # then look at distribution across bouts, across files (as some files don't have bouts)
            bout_disp_bps.append(bout_disp_all)
            # bout duration has been calculated before, no need to include
            # bout_duration_bps.append(bout_duration_all)
            # average speed might get inflated due to short bouts, not accurate measurement
            # bout_avg_speed_bps.append(bout_avg_speed_all)
            # behavior_all_bps.append(np.hstack(behavior_all))

        behavioral_sums = {key: [] for key in st.session_state["annotated_behaviors"]}

        # look at speed for each behavior
        for b, behav in enumerate(st.session_state['annotated_behaviors']):
            behavioral_sums[behav] = np.hstack([np.hstack(
                [bout_disp_bps[bp][f][behav]
                 for f in range(len(bout_disp_bps[bp]))])
                for bp in range(len(bout_disp_bps))])
        max_perb = []
        for beh in list(behavioral_sums.keys()):
            if len(behavioral_sums[beh]) > 10:
                max_perb.append(np.percentile(behavioral_sums[beh], 95))
        max_all = np.max(max_perb)
        movement_n_bins = st.slider('number of movement bins',
                                    min_value=5,
                                    max_value=30,
                                    value=10, key=f'move_bin_slider_{condition}')
        pre_alloc_movement = np.zeros((len(list(behavioral_sums.keys())), movement_n_bins))
        label_encoder = LabelEncoder()
        for b, behav in enumerate(list(behavioral_sums.keys())):
            df = pd.DataFrame(data=behavioral_sums[behav],
                              columns=['bp_movement'])
            n_bins = np.linspace(0, max_all, movement_n_bins)
            y = label_encoder.fit_transform(pd.cut(df['bp_movement'], n_bins, retbins=True)[0])
            pre_alloc_movement[b, :] = np.histogram(y, bins=np.arange(0, movement_n_bins+1))[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_sums = pre_alloc_movement.sum(axis=1)
            movement_by_behav = pre_alloc_movement / row_sums[:, np.newaxis]
        fig = px.imshow(movement_by_behav,
                        color_continuous_scale='agsunset',
                        aspect='equal'
                        )
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(0, movement_by_behav.shape[0]),
                ticktext=st.session_state['annotated_behaviors']
            ),
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(0, movement_by_behav.shape[1]),
                ticktext=np.round(n_bins, 1)
            )
        )
        fig.update_layout(xaxis=dict(title='Avg. movement per bout (Δpixels)'))
        fig.layout.coloraxis.showscale = False
        st.plotly_chart(fig, use_container_width=True)


        # fig = go.Figure()
        # for b, behav in enumerate(st.session_state['annotated_behaviors']):
        #     # st.write(b, behav)
        #     try:
        #         # remove outliers
        #         # fig.add_trace(go.Box(
        #         #     y=behavioral_speed[st.session_state['annotated_behaviors'][behav]]
        #         #     [(behavioral_speed[st.session_state['annotated_behaviors'][behav]] <
        #         #       np.percentile(behavioral_speed[st.session_state['annotated_behaviors'][behav]], 95)) &
        #         #      (behavioral_speed[st.session_state['annotated_behaviors'][behav]] >
        #         #       np.percentile(behavioral_speed[st.session_state['annotated_behaviors'][behav]], 5))],
        #         #     name=st.session_state['annotated_behaviors'][behav],
        #         #     line_color=behavior_colors[st.session_state['annotated_behaviors'][behav]],
        #         #     boxpoints=False,
        #         # ))
        #         fig.add_trace(go.Box(
        #             x=behavioral_sums[behav]
        #             [(behavioral_sums[behav] <
        #               np.percentile(behavioral_sums[behav], 97.5)) &
        #              (behavioral_sums[behav] >
        #               np.percentile(behavioral_sums[behav], 2.5))],
        #             name=behav,
        #             line_color=behavior_colors[behav],
        #             boxpoints=False,
        #         ))
        #     except:
        #         pass
        # # fig.update_layout(yaxis=dict(title='bout movement speed (Δpixels/second)'),
        # #                   )
        # fig.update_layout(xaxis=dict(title='Avg. movement per bout (Δpixels)'))
        # fig.update_traces(orientation='h')
        # fig['layout']['yaxis']['autorange'] = "reversed"
        #
        # st.plotly_chart(fig, use_container_width=True)


def condition_kinematix_plot():
    figure_container = st.container()
    rows = int(np.ceil(st.session_state['num_condition'] / 2))
    mod_ = st.session_state['num_condition'] % 2
    count = 0
    for row in range(rows):
        left_col, right_col = figure_container.columns(2)
        # left stays
        left_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                           f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 1}</h1> "
                           , unsafe_allow_html=True)
        if f'kine_pred_{count + 1}' not in st.session_state:
            st.session_state[f'kine_pred_{count + 1}'] = None
        kinematix_predict(left_col,
                          count + 1,
                          st.session_state[f'feats_condition_{count + 1}'],
                          st.session_state[f'filt_pose_condition_{count + 1}'],
                          st.session_state['behavior_colors'])

        # ridge_csv = kinematics_csv(
        #     list(st.session_state['features'].keys())[count],
        # )
        # left_expander.download_button(
        #     label="Download data as CSV",
        #     data=ridge_csv,
        #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
        #     mime='text/csv',
        #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
        # )
        count += 1
        # right only when multiples of 2 or
        if row == rows - 1:
            if mod_ == 0:
                right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                                   f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                                   , unsafe_allow_html=True)
                if f'kine_pred_{count + 1}' not in st.session_state:
                    st.session_state[f'kine_pred_{count + 1}'] = None
                kinematix_predict(right_col,
                                  count + 1,
                                  st.session_state[f'feats_condition_{count + 1}'],
                                  st.session_state[f'filt_pose_condition_{count + 1}'],
                                  st.session_state['behavior_colors'])
                # ridge_csv = kinematics_csv(
                #     list(st.session_state['features'].keys())[count],
                # )
                # right_expander.download_button(
                #     label="Download data as CSV",
                #     data=ridge_csv,
                #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
                #     mime='text/csv',
                #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
                # )
                count += 1
        else:
            right_col.markdown(f" <h1 style='text-align: left; color: #EEEEEE; font-size:16px; "
                               f"font-family:Avenir; font-weight:normal'>Condition {row * 2 + 2}</h1> "
                               , unsafe_allow_html=True)
            if f'kine_pred_{count + 1}' not in st.session_state:
                st.session_state[f'kine_pred_{count + 1}'] = None
            kinematix_predict(right_col,
                              count + 1,
                              st.session_state[f'feats_condition_{count + 1}'],
                              st.session_state[f'filt_pose_condition_{count + 1}'],
                              st.session_state['behavior_colors'])
            # ridge_csv = kinematics_csv(
            #     list(st.session_state['features'].keys())[count],
            # )
            # right_expander.download_button(
            #     label="Download data as CSV",
            #     data=ridge_csv,
            #     file_name=f"{list(st.session_state['features'].keys())[count]}.csv",
            #     mime='text/csv',
            #     key=f"{list(st.session_state['features'].keys())[count]}_dwnload"
            # )
            count += 1
