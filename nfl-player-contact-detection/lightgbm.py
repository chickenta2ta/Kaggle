import gc
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PATH_TO_INPUT = "../input/nfl-player-contact-detection"
PATH_TO_OUTPUT = "."

TRAIN_LABELS = "train_labels.csv"
TRAIN_PLAYER_TRACKING = "train_player_tracking.csv"
TRAIN_BASELINE_HELMETS = "train_baseline_helmets.csv"
TRAIN_VIDEO_METADATA = "train_video_metadata.csv"

SAMPLE_SUBMISSION = "sample_submission.csv"
TEST_PLAYER_TRACKING = "test_player_tracking.csv"
TEST_BASELINE_HELMETS = "test_baseline_helmets.csv"
TEST_VIDEO_METADATA = "test_video_metadata.csv"

SUBMISSION = "submission.csv"

train_labels = pd.read_csv(os.path.join(PATH_TO_INPUT, TRAIN_LABELS))
train_player_tracking = pd.read_csv(os.path.join(PATH_TO_INPUT, TRAIN_PLAYER_TRACKING))
train_baseline_helmets = pd.read_csv(
    os.path.join(PATH_TO_INPUT, TRAIN_BASELINE_HELMETS)
)
train_video_metadata = pd.read_csv(os.path.join(PATH_TO_INPUT, TRAIN_VIDEO_METADATA))

sample_submission = pd.read_csv(os.path.join(PATH_TO_INPUT, SAMPLE_SUBMISSION))
test_player_tracking = pd.read_csv(os.path.join(PATH_TO_INPUT, TEST_PLAYER_TRACKING))
test_baseline_helmets = pd.read_csv(os.path.join(PATH_TO_INPUT, TEST_BASELINE_HELMETS))
test_video_metadata = pd.read_csv(os.path.join(PATH_TO_INPUT, TEST_VIDEO_METADATA))

# Endzone2 should be ignored as it is a merging error or something
train_baseline_helmets = train_baseline_helmets[
    train_baseline_helmets["view"] != "Endzone2"
]


def join_baseline_helmets_to_labels(
    labels,
    baseline_helmets,
    video_metadata,
    baseline_helmets_columns,
    fps=59.94,
):
    video_metadata = video_metadata[["game_play", "view", "start_time"]]
    baseline_helmets = baseline_helmets.merge(
        video_metadata, how="left", on=["game_play", "view"]
    )

    baseline_helmets["frame_time"] = pd.to_datetime(
        baseline_helmets["start_time"]
    ) + pd.to_timedelta(baseline_helmets["frame"] / fps, unit="S")
    baseline_helmets["frame_time"] = baseline_helmets["frame_time"].dt.round("100L")

    baseline_helmets = baseline_helmets.groupby(
        ["game_play", "view", "nfl_player_id", "frame_time"], as_index=False
    ).agg({column_name: "mean" for column_name in baseline_helmets_columns})

    labels["datetime"] = pd.to_datetime(labels["datetime"])
    baseline_helmets = baseline_helmets.astype({"nfl_player_id": "string"})

    for view in ["Sideline", "Endzone"]:
        view_lower = view.lower()

        baseline_helmets_view = baseline_helmets[baseline_helmets["view"] == view]

        for i in range(1, 3):
            columns = {
                column_name: column_name + f"_{view_lower}_{i}"
                for column_name in baseline_helmets_columns
            }

            labels = labels.merge(
                baseline_helmets_view,
                how="left",
                left_on=["game_play", f"nfl_player_id_{i}", "datetime"],
                right_on=["game_play", "nfl_player_id", "frame_time"],
            )
            labels.rename(columns=columns, inplace=True)
            labels.drop(columns=["view", "nfl_player_id", "frame_time"], inplace=True)

    return labels


def join_player_tracking_and_baseline_helmets_to_labels(
    labels,
    player_tracking,
    baseline_helmets,
    video_metadata,
    player_tracking_columns=[
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ],
    baseline_helmets_columns=["left", "width", "top", "height"],
):
    labels = labels.astype(
        {
            "step": "string",
            "nfl_player_id_1": "string",
            "nfl_player_id_2": "string",
        }
    )
    player_tracking = player_tracking.astype(
        {"nfl_player_id": "string", "step": "string"}
    )

    player_tracking = player_tracking[
        ["game_play", "nfl_player_id", "step"] + player_tracking_columns
    ]

    for i in range(1, 3):
        columns = {
            column_name: column_name + f"_{i}"
            for column_name in player_tracking_columns
        }

        labels = labels.merge(
            player_tracking,
            how="left",
            left_on=["game_play", "step", f"nfl_player_id_{i}"],
            right_on=["game_play", "step", "nfl_player_id"],
        )
        labels.rename(columns=columns, inplace=True)
        labels.drop(columns="nfl_player_id", inplace=True)

    labels = join_baseline_helmets_to_labels(
        labels, baseline_helmets, video_metadata, baseline_helmets_columns
    )

    # This column is necessary to split data in a stratified fashion
    labels["g_flag"] = labels["nfl_player_id_2"] == "G"

    return labels


def separate_player_and_ground(labels):
    is_ground = labels["nfl_player_id_2"] == "G"

    labels_player = labels[~is_ground].copy()
    labels_ground = labels[is_ground].copy()

    return labels_player, labels_ground


def append_1(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_1")

    return appended_column_names


def append_1_and_2(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names += [column_name + "_1", column_name + "_2"]

    return appended_column_names


def append_sideline(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_sideline")

    return appended_column_names


def append_endzone(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_endzone")

    return appended_column_names


def append_sideline_and_endzone(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names += [column_name + "_sideline", column_name + "_endzone"]

    return appended_column_names


def append_product(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_product")

    return appended_column_names


def append_difference(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_difference")

    return appended_column_names


def column_exists(labels, column_names):
    for column_name in column_names:
        if column_name not in labels.columns.to_list():
            return False

    return True


def calculate_iou(labels):
    feature_columns = []
    for view in ["sideline", "endzone"]:
        labels[f"right_{view}_1"] = labels[f"left_{view}_1"] + labels[f"width_{view}_1"]
        labels[f"right_{view}_2"] = labels[f"left_{view}_2"] + labels[f"width_{view}_2"]
        labels[f"dx_{view}"] = labels[[f"right_{view}_1", f"right_{view}_2"]].min(
            axis=1, skipna=False
        ) - labels[[f"left_{view}_1", f"left_{view}_2"]].max(axis=1, skipna=False)

        labels[f"bottom_{view}_1"] = (
            labels[f"top_{view}_1"] + labels[f"height_{view}_1"]
        )
        labels[f"bottom_{view}_2"] = (
            labels[f"top_{view}_2"] + labels[f"height_{view}_2"]
        )
        labels[f"dy_{view}"] = labels[[f"bottom_{view}_1", f"bottom_{view}_2"]].min(
            axis=1, skipna=False
        ) - labels[[f"top_{view}_1", f"top_{view}_2"]].max(axis=1, skipna=False)

        has_intersection = (labels[f"dx_{view}"] > 0) & (labels[f"dy_{view}"] > 0)

        area_of_intersection = labels[f"dx_{view}"] * labels[f"dy_{view}"]
        area_of_intersection[~has_intersection] = 0

        area_of_union = (
            (labels[f"height_{view}_1"] * labels[f"width_{view}_1"])
            + (labels[f"height_{view}_2"] * labels[f"width_{view}_2"])
            - area_of_intersection
        )

        labels[f"iou_{view}"] = area_of_intersection / area_of_union
        labels.fillna(value={f"iou_{view}": 0})

        labels.drop(
            columns=[
                f"right_{view}_1",
                f"right_{view}_2",
                f"dx_{view}",
                f"bottom_{view}_1",
                f"bottom_{view}_2",
                f"dy_{view}",
            ],
            inplace=True,
        )

        feature_columns.append(f"iou_{view}")

    return labels, feature_columns


def calculate_moving_average(labels, column_names, windows=[3, 5, 10]):
    feature_columns = []

    labels.sort_values(
        by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
    )

    for column_name in column_names:
        for window in windows:
            labels[column_name + f"_moving_average_{window}"] = (
                labels.groupby(
                    ["game_play", "nfl_player_id_1", "nfl_player_id_2"],
                    as_index=False,
                    sort=False,
                )[column_name]
                .rolling(window)
                .mean()[column_name]
            )

            feature_columns.append(column_name + f"_moving_average_{window}")

    labels.sort_index(inplace=True)

    return labels, feature_columns


def create_features_for_player(
    labels,
    append_1_and_2_columns=["x_position", "y_position", "speed", "sa"],
    append_sideline_1_and_2_columns=["left", "width", "top", "height"],
    append_endzone_1_and_2_columns=["width", "top"],
    product_columns=["speed", "distance", "acceleration"],
    player_tracking_difference_columns=["speed", "direction", "orientation"],
    baseline_helmets_difference_columns=["left", "width", "top", "height"],
    sum_columns=["sa", "iou"],
    append_endzone_1_and_2_moving_average_columns=["top"],
    product_moving_average_columns=["speed", "acceleration"],
    player_tracking_difference_moving_average_columns=["direction", "orientation"],
    baseline_helmets_difference_moving_average_columns=["left", "top"],
    is_training=True,
):
    feature_columns = []

    necessary_columns = ["x_position", "y_position"]
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels["distance"] = np.sqrt(
            ((labels["x_position_1"] - labels["x_position_2"]) ** 2)
            + ((labels["y_position_1"] - labels["y_position_2"]) ** 2)
        )
        feature_columns.append("distance")

    if is_training:
        is_close = get_indices_of_closer_than_threshold(labels)
        labels = labels[is_close].copy()
        labels.reset_index(drop=True, inplace=True)

        del is_close
        gc.collect()

    necessary_columns = append_1_and_2(append_1_and_2_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    necessary_columns = append_sideline(append_sideline_1_and_2_columns)
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    necessary_columns = append_endzone(append_endzone_1_and_2_columns)
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    necessary_columns = ["left", "width", "top", "height"]
    necessary_columns = append_sideline_and_endzone(necessary_columns)
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_iou(labels)
        feature_columns += columns

    # Add product features
    for column_name in product_columns:
        necessary_columns = [column_name]
        necessary_columns = append_1_and_2(necessary_columns)
        if column_exists(labels, necessary_columns):
            labels[column_name + "_product"] = (
                labels[column_name + "_1"] * labels[column_name + "_2"]
            )
            feature_columns.append(column_name + "_product")

    # Add difference features
    for column_name in player_tracking_difference_columns:
        necessary_columns = [column_name]
        necessary_columns = append_1_and_2(necessary_columns)
        if column_exists(labels, necessary_columns):
            labels[column_name + "_difference"] = abs(
                labels[column_name + "_1"] - labels[column_name + "_2"]
            )
            feature_columns.append(column_name + "_difference")

    for column_name in baseline_helmets_difference_columns:
        necessary_columns = [column_name]
        necessary_columns = append_sideline_and_endzone(necessary_columns)
        necessary_columns = append_1_and_2(necessary_columns)
        if column_exists(labels, necessary_columns):
            labels[column_name + "_sideline_difference"] = abs(
                labels[column_name + "_sideline_1"]
                - labels[column_name + "_sideline_2"]
            )
            labels[column_name + "_endzone_difference"] = abs(
                labels[column_name + "_endzone_1"] - labels[column_name + "_endzone_2"]
            )
            feature_columns += [
                column_name + "_sideline_difference",
                column_name + "_endzone_difference",
            ]

    # Add sum features
    for column_name in sum_columns:
        necessary_columns = [column_name]

        if column_exists(labels, append_1_and_2(necessary_columns)):
            labels[column_name + "_sum"] = (
                labels[column_name + "_1"] + labels[column_name + "_2"]
            )
            feature_columns.append(column_name + "_sum")

        if column_exists(labels, append_sideline_and_endzone(necessary_columns)):
            labels[column_name + "_sum"] = (
                labels[column_name + "_sideline"] + labels[column_name + "_endzone"]
            )
            feature_columns.append(column_name + "_sum")

    # Add moving average features
    necessary_columns = ["distance"]
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_endzone(append_endzone_1_and_2_moving_average_columns)
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_product(product_moving_average_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_difference(
        player_tracking_difference_moving_average_columns
    )
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_sideline_and_endzone(
        baseline_helmets_difference_moving_average_columns
    )
    necessary_columns = append_difference(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    return labels, feature_columns


def create_features_for_ground(
    labels,
    append_1_columns=[
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ],
    append_sideline_1_columns=["left", "width", "top", "height"],
    append_endzone_1_columns=["width", "top"],
    append_1_moving_average_columns=[
        "x_position",
        "y_position",
        "speed",
        "orientation",
        "sa",
    ],
    append_sideline_1_moving_average_columns=["left", "top", "height"],
    append_endzone_1_moving_average_columns=["width", "top"],
):
    feature_columns = []

    necessary_columns = append_1(append_1_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    necessary_columns = append_sideline(append_sideline_1_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    necessary_columns = append_endzone(append_endzone_1_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        feature_columns += necessary_columns

    # Add moving average features
    necessary_columns = append_1(append_1_moving_average_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_sideline(append_sideline_1_moving_average_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    necessary_columns = append_endzone(append_endzone_1_moving_average_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(labels, necessary_columns)
        feature_columns += columns

    return labels, feature_columns


def get_indices_of_closer_than_threshold(labels, threshold=2):
    necessary_columns = ["distance"]
    if column_exists(labels, necessary_columns):
        is_close = (labels["distance"] <= threshold) | labels["distance"].isnull()
        return is_close
    else:
        return pd.Series(True, index=labels.index)


def train_lightgbm(
    X_train,
    y_train,
    param={
        "objective": "binary",
        "num_boost_round": 10_000,
        "learning_rate": 0.03,
        "device_type": "gpu" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "force_row_wise": True,
        "early_stopping_round": 100,
        "metric": "auc",
    },
):
    skf = StratifiedKFold(shuffle=True, random_state=42)

    # This weird implementation is to avoid LightGBM warnings
    num_boost_round = param["num_boost_round"]
    stopping_rounds = param["early_stopping_round"]

    param_copy = param.copy()
    del param_copy["num_boost_round"], param_copy["early_stopping_round"]

    models = []
    for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {i + 1}:")

        X_train_fold, y_train_fold = X_train.loc[train_index], y_train[train_index]
        X_test_fold, y_test_fold = X_train.loc[test_index], y_train[test_index]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        validation_data = lgb.Dataset(
            X_test_fold, label=y_test_fold, reference=train_data
        )

        model = lgb.train(
            param_copy,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[validation_data],
            callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds)],
        )

        y_pred = model.predict(X_test_fold, num_iteration=model.best_iteration)
        score = roc_auc_score(y_test_fold, y_pred)
        print(f"Score: {score}")

        models.append(model)

    return models


def split_contact_id(sample_submission):
    sample_submission[
        ["game", "play", "step", "nfl_player_id_1", "nfl_player_id_2"]
    ] = sample_submission["contact_id"].str.split("_", expand=True)

    sample_submission["game_play"] = sample_submission["game"].str.cat(
        sample_submission["play"], sep="_"
    )
    sample_submission.drop(columns=["game", "play"], inplace=True)

    return sample_submission


def join_datetime_to_labels(sample_submission, player_tracking):
    player_tracking = player_tracking[["game_play", "datetime", "step"]]

    player_tracking = player_tracking[
        ~player_tracking.duplicated(subset=["game_play", "step"])
    ]

    sample_submission = sample_submission.astype({"step": "string"})
    player_tracking = player_tracking.astype({"step": "string"})

    sample_submission = sample_submission.merge(
        player_tracking, how="left", on=["game_play", "step"]
    )

    return sample_submission


def matthews_corrcoef_(x, y_train, y_pred_train):
    mcc = matthews_corrcoef(y_train, y_pred_train > x[0])
    return -mcc


def predict_lightgbm(
    models,
    X_train,
    y_train,
    X_test,
    x0=[0.5],
):
    y_pred_train = np.zeros(len(X_train))
    y_pred_test = np.zeros(len(X_test))

    for model in models:
        y_pred_train += model.predict(
            X_train, num_iteration=model.best_iteration
        ) / len(models)
        y_pred_test += model.predict(X_test, num_iteration=model.best_iteration) / len(
            models
        )

    res = minimize(
        matthews_corrcoef_, x0, args=(y_train, y_pred_train), method="Nelder-Mead"
    )
    y_test = (y_pred_test > res.x[0]).astype("int")

    return y_test


def predict_on_test_data(
    models_player,
    models_ground,
    train_labels_player,
    train_labels_ground,
    test_labels,
    feature_columns_player,
    feature_columns_ground,
    is_submission=True,
):
    test_labels_player, test_labels_ground = separate_player_and_ground(test_labels)

    # Do not reset index. We rather want to keep the order of sample_submission.csv
    # test_labels_player.reset_index(drop=True, inplace=True)
    # test_labels_ground.reset_index(drop=True, inplace=True)

    test_labels_player, _ = create_features_for_player(
        test_labels_player, is_training=False
    )
    test_labels_ground, _ = create_features_for_ground(test_labels_ground)

    y_name = "contact"
    if not is_submission:
        y_name += "_pred"

    test_labels_player[y_name] = predict_lightgbm(
        models_player,
        train_labels_player[feature_columns_player],
        train_labels_player["contact"],
        test_labels_player[feature_columns_player],
    )
    test_labels_ground[y_name] = predict_lightgbm(
        models_ground,
        train_labels_ground[feature_columns_ground],
        train_labels_ground["contact"],
        test_labels_ground[feature_columns_ground],
    )

    test_labels = pd.concat([test_labels_player, test_labels_ground])
    test_labels.sort_index(inplace=True)

    is_close = get_indices_of_closer_than_threshold(test_labels)
    test_labels.loc[~is_close, y_name] = 0

    if is_submission:
        test_labels[["contact_id", "contact"]].to_csv(
            os.path.join(PATH_TO_OUTPUT, SUBMISSION), index=False
        )
    else:
        mcc = matthews_corrcoef(test_labels["contact"], test_labels["contact_pred"])
        print(f"MCC: {mcc}")


train_labels = join_player_tracking_and_baseline_helmets_to_labels(
    train_labels, train_player_tracking, train_baseline_helmets, train_video_metadata
)

del train_player_tracking, train_baseline_helmets, train_video_metadata
gc.collect()

train_labels, test_labels = train_test_split(
    train_labels, random_state=42, stratify=train_labels[["contact", "g_flag"]]
)
train_labels.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)

(
    train_labels_player,
    train_labels_ground,
) = separate_player_and_ground(train_labels)
train_labels_player.reset_index(drop=True, inplace=True)
train_labels_ground.reset_index(drop=True, inplace=True)

del train_labels
gc.collect()

train_labels_player, feature_columns_player = create_features_for_player(
    train_labels_player
)
train_labels_ground, feature_columns_ground = create_features_for_ground(
    train_labels_ground
)

models_player = train_lightgbm(
    train_labels_player[feature_columns_player], train_labels_player["contact"]
)
models_ground = train_lightgbm(
    train_labels_ground[feature_columns_ground], train_labels_ground["contact"]
)

sample_submission = split_contact_id(sample_submission)
sample_submission = join_datetime_to_labels(sample_submission, test_player_tracking)

sample_submission = join_player_tracking_and_baseline_helmets_to_labels(
    sample_submission, test_player_tracking, test_baseline_helmets, test_video_metadata
)

predict_on_test_data(
    models_player,
    models_ground,
    train_labels_player,
    train_labels_ground,
    sample_submission,
    feature_columns_player,
    feature_columns_ground,
)

# Evaluate the model before submission
predict_on_test_data(
    models_player,
    models_ground,
    train_labels_player,
    train_labels_ground,
    test_labels,
    feature_columns_player,
    feature_columns_ground,
    is_submission=False,
)

# Visualize feature importance
fig, axs = plt.subplots(2, figsize=(6.4 * 5, 4.8 * 5))
lgb.plot_importance(
    models_player[0],
    ax=axs[0],
    title="Feature importance of player",
)
lgb.plot_importance(
    models_ground[0],
    ax=axs[1],
    title="Feature importance of ground",
)
plt.show()
