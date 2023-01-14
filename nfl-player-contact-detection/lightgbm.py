import gc
import os

import lightgbm as lgb
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

    labels["g_flag"] = labels["nfl_player_id_2"] == "G"
    labels = labels.astype({"g_flag": "int"})
    feature_columns.append("g_flag")

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


train_labels, feature_columns = create_features(
    train_labels, train_player_tracking, train_baseline_helmets, train_video_metadata
)

DISTANCE_THRESHOLD = 2

if "distance" in feature_columns:
    train_labels = train_labels[
        (train_labels["distance"] <= DISTANCE_THRESHOLD)
        | (train_labels["distance"].isnull())
    ]
    train_labels.reset_index(drop=True, inplace=True)

train_labels, columns = add_product_and_difference_features(train_labels)
feature_columns += columns

X_train = train_labels[feature_columns]
y_train = train_labels["contact"]

param = {
    "objective": "binary",
    "device_type": "gpu" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "force_row_wise": True,
    "metric": "auc",
}

skf = StratifiedKFold(shuffle=True, random_state=42)

models = []
scores = []
for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {i + 1}:")

    X_train_fold, y_train_fold = X_train.loc[train_index], y_train[train_index]
    X_test_fold, y_test_fold = X_train.loc[test_index], y_train[test_index]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    validation_data = lgb.Dataset(X_test_fold, label=y_test_fold, reference=train_data)

    model = lgb.train(
        param,
        train_data,
        num_boost_round=20_000,
        valid_sets=[validation_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
    )

    y_pred = model.predict(X_test_fold, num_iteration=model.best_iteration)
    score = roc_auc_score(y_test_fold, y_pred)

    models.append(model)
    scores.append(score)

del (
    X_train_fold,
    train_baseline_helmets,
    train_index,
    train_labels,
    train_player_tracking,
    train_video_metadata,
    y_pred,
    y_test_fold,
    y_train_fold,
)
gc.collect()

sample_submission = split_contact_id(sample_submission)
sample_submission = join_datetime_to_labels(sample_submission, test_player_tracking)

sample_submission, feature_columns = create_features(
    sample_submission, test_player_tracking, test_baseline_helmets, test_video_metadata
)

sample_submission, columns = add_product_and_difference_features(sample_submission)
feature_columns += columns

X_test = sample_submission[feature_columns]
y_pred_train = np.zeros(len(X_train))
y_pred_test = np.zeros(len(X_test))

for model in models:
    y_pred_train += model.predict(X_train, num_iteration=model.best_iteration) / len(
        models
    )
    y_pred_test += model.predict(X_test, num_iteration=model.best_iteration) / len(
        models
    )

x0 = [0.5]
res = minimize(matthews_corrcoef_, x0, args=(y_train, y_pred_train))

sample_submission["contact"] = (y_pred_test > res.x[0]).astype("int")
sample_submission.loc[sample_submission["distance"] > DISTANCE_THRESHOLD, "contact"] = 0
sample_submission[["contact_id", "contact"]].to_csv(
    os.path.join(PATH_TO_OUTPUT, SUBMISSION), index=False
)
