import gc
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

    feature_columns = []
    for i in range(1, 3):
        for view in ["Sideline", "Endzone"]:
            view_lower = view.lower()
            columns = {
                column_name: column_name + f"_{i}_{view_lower}"
                for column_name in baseline_helmets_columns
            }

            baseline_helmets_view = baseline_helmets[baseline_helmets["view"] == view]

            labels = labels.merge(
                baseline_helmets_view,
                how="left",
                left_on=["game_play", f"nfl_player_id_{i}", "datetime"],
                right_on=["game_play", "nfl_player_id", "frame_time"],
            )
            labels.rename(columns=columns, inplace=True)
            labels.drop(columns=["view", "nfl_player_id", "frame_time"], inplace=True)

            feature_columns += columns.values()

    return labels, feature_columns


def calculate_iou(labels):
    feature_columns = []
    for view in ["sideline", "endzone"]:
        labels[f"right_1_{view}"] = labels[f"left_1_{view}"] + labels[f"width_1_{view}"]
        labels[f"right_2_{view}"] = labels[f"left_2_{view}"] + labels[f"width_2_{view}"]
        labels[f"dx_{view}"] = labels[[f"right_1_{view}", f"right_2_{view}"]].min(
            axis=1, skipna=False
        ) - labels[[f"left_1_{view}", f"left_2_{view}"]].max(axis=1, skipna=False)

        labels[f"bottom_1_{view}"] = (
            labels[f"top_1_{view}"] + labels[f"height_1_{view}"]
        )
        labels[f"bottom_2_{view}"] = (
            labels[f"top_2_{view}"] + labels[f"height_2_{view}"]
        )
        labels[f"dy_{view}"] = labels[[f"bottom_1_{view}", f"bottom_2_{view}"]].min(
            axis=1, skipna=False
        ) - labels[[f"top_1_{view}", f"top_2_{view}"]].max(axis=1, skipna=False)

        has_intersection = (labels[f"dx_{view}"] > 0) & (labels[f"dy_{view}"] > 0)

        area_of_intersection = labels[f"dx_{view}"] * labels[f"dy_{view}"]
        area_of_intersection[~has_intersection] = 0

        area_of_union = (
            (labels[f"height_1_{view}"] * labels[f"width_1_{view}"])
            + (labels[f"height_2_{view}"] * labels[f"width_2_{view}"])
            - area_of_intersection
        )

        labels[f"iou_{view}"] = area_of_intersection / area_of_union
        labels.fillna(value={f"iou_{view}": 0})

        labels.drop(
            columns=[
                f"right_1_{view}",
                f"right_2_{view}",
                f"dx_{view}",
                f"bottom_1_{view}",
                f"bottom_2_{view}",
                f"dy_{view}",
            ],
            inplace=True,
        )

        feature_columns.append(f"iou_{view}")

    return labels, feature_columns


def create_features(
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

    feature_columns = []
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

        feature_columns += columns.values()

    if (
        "x_position" in player_tracking_columns
        and "y_position" in player_tracking_columns
    ):
        labels["distance"] = np.sqrt(
            ((labels["x_position_1"] - labels["x_position_2"]) ** 2)
            + ((labels["y_position_1"] - labels["y_position_2"]) ** 2)
        )
        feature_columns.append("distance")

    labels, columns = join_baseline_helmets_to_labels(
        labels, baseline_helmets, video_metadata, baseline_helmets_columns
    )
    feature_columns += columns

    labels, columns = calculate_iou(labels)
    feature_columns += columns

    return labels, feature_columns


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


def add_product_and_difference_features(labels):
    feature_columns = []
    for column_name in [
        "x_position",
        "y_position",
        "speed",
        "distance",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]:
        column_name_1 = column_name + "_1"
        column_name_2 = column_name + "_2"

        if (
            column_name_1 in labels.columns.to_list()
            and column_name_2 in labels.columns.to_list()
        ):
            column_name_product = column_name + "_product"
            labels[column_name_product] = labels[column_name_1] * labels[column_name_2]

            column_name_difference = column_name + "_difference"
            labels[column_name_difference] = abs(
                labels[column_name_1] - labels[column_name_2]
            )

            feature_columns += [column_name_product, column_name_difference]

    for column_name in [
        "iou",
    ]:
        column_name_sideline = column_name + "_sideline"
        column_name_endzone = column_name + "_endzone"

        if (
            column_name_sideline in labels.columns.to_list()
            and column_name_endzone in labels.columns.to_list()
        ):
            column_name_product = column_name + "_product"
            labels[column_name_product] = (
                labels[column_name_sideline] * labels[column_name_endzone]
            )

            column_name_difference = column_name + "_difference"
            labels[column_name_difference] = (
                labels[column_name_sideline] - labels[column_name_endzone]
            )

            feature_columns += [column_name_product, column_name_difference]

    return labels, feature_columns


def separate_player_and_ground(labels, feature_columns_player):
    is_ground = labels["nfl_player_id_2"] == "G"

    labels_player = labels[~is_ground].copy()
    labels_ground = labels[is_ground].copy()

    feature_columns_ground = []
    for column_name in feature_columns_player:
        if (
            "_2" in column_name
            or column_name == "distance"
            or column_name.startswith("iou_")
        ):
            continue
        else:
            feature_columns_ground.append(column_name)

    return labels_player, labels_ground, feature_columns_ground


def get_indices_of_closer_than_threshold(labels, feature_columns, threshold=2):
    if "distance" in feature_columns:
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


train_labels, feature_columns_player = create_features(
    train_labels, train_player_tracking, train_baseline_helmets, train_video_metadata
)

is_close = get_indices_of_closer_than_threshold(train_labels, feature_columns_player)
train_labels = train_labels[is_close]
train_labels.reset_index(drop=True, inplace=True)

(
    train_labels_player,
    train_labels_ground,
    feature_columns_ground,
) = separate_player_and_ground(train_labels, feature_columns_player)
train_labels_player.reset_index(drop=True, inplace=True)
train_labels_ground.reset_index(drop=True, inplace=True)

del train_labels
gc.collect()

train_labels_player, columns = add_product_and_difference_features(train_labels_player)
feature_columns_player += columns

models_player = train_lightgbm(
    train_labels_player[feature_columns_player], train_labels_player["contact"]
)
models_ground = train_lightgbm(
    train_labels_ground[feature_columns_ground], train_labels_ground["contact"]
)

del (
    train_player_tracking,
    train_baseline_helmets,
    train_video_metadata,
    is_close,
)
gc.collect()

sample_submission = split_contact_id(sample_submission)
sample_submission = join_datetime_to_labels(sample_submission, test_player_tracking)

sample_submission, feature_columns_player = create_features(
    sample_submission, test_player_tracking, test_baseline_helmets, test_video_metadata
)

(
    sample_submission_player,
    sample_submission_ground,
    feature_columns_ground,
) = separate_player_and_ground(sample_submission, feature_columns_player)

# Do not reset index. We rather want to keep the order of sample_submission.csv
# sample_submission_player.reset_index(drop=True, inplace=True)
# sample_submission_ground.reset_index(drop=True, inplace=True)

sample_submission_player, columns = add_product_and_difference_features(
    sample_submission_player
)
feature_columns_player += columns

sample_submission_player["contact"] = predict_lightgbm(
    models_player,
    train_labels_player[feature_columns_player],
    train_labels_player["contact"],
    sample_submission_player[feature_columns_player],
)
sample_submission_ground["contact"] = predict_lightgbm(
    models_ground,
    train_labels_ground[feature_columns_ground],
    train_labels_ground["contact"],
    sample_submission_ground[feature_columns_ground],
)

sample_submission = pd.concat([sample_submission_player, sample_submission_ground])
sample_submission.sort_index(inplace=True)

is_close = get_indices_of_closer_than_threshold(
    sample_submission, feature_columns_player
)
sample_submission.loc[~is_close, "contact"] = 0

sample_submission[["contact_id", "contact"]].to_csv(
    os.path.join(PATH_TO_OUTPUT, SUBMISSION), index=False
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
