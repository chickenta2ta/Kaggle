import os

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
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

    baseline_helmets = baseline_helmets[
        ["game_play", "view", "nfl_player_id", "frame_time"] + baseline_helmets_columns
    ]

    labels["datetime"] = pd.to_datetime(labels["datetime"])

    datetime_to_frame_time = []
    for game_play in labels["game_play"].unique():
        labels_game_play = labels[labels["game_play"] == game_play].copy()
        baseline_helmets_game_play = baseline_helmets[
            baseline_helmets["game_play"] == game_play
        ].copy()

        labels_game_play = labels_game_play["datetime"]
        baseline_helmets_game_play = baseline_helmets_game_play["frame_time"]

        labels_game_play = labels_game_play.drop_duplicates()
        baseline_helmets_game_play = baseline_helmets_game_play.drop_duplicates()

        labels_game_play.sort_values(inplace=True)
        baseline_helmets_game_play.sort_values(inplace=True)

        merged = pd.merge_asof(
            labels_game_play,
            baseline_helmets_game_play,
            left_on="datetime",
            right_on="frame_time",
            direction="nearest",
        )
        merged["game_play"] = game_play

        datetime_to_frame_time.append(merged)

    datetime_to_frame_time = pd.concat(datetime_to_frame_time)

    labels = labels.merge(
        datetime_to_frame_time, how="left", on=["game_play", "datetime"]
    )

    baseline_helmets = baseline_helmets.merge(
        datetime_to_frame_time, how="right", on=["game_play", "frame_time"]
    )
    baseline_helmets.drop(columns="datetime", inplace=True)

    # Make the combination of game_play, view, nfl_player_id and frame_time unique
    baseline_helmets = baseline_helmets[
        ~baseline_helmets.duplicated(
            subset=["game_play", "view", "nfl_player_id", "frame_time"]
        )
    ]

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
                left_on=["game_play", f"nfl_player_id_{i}", "frame_time"],
                right_on=["game_play", "nfl_player_id", "frame_time"],
            )
            labels.rename(columns=columns, inplace=True)
            labels.drop(columns=["view", "nfl_player_id"], inplace=True)

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
