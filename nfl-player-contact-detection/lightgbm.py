import os

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
    baseline_helmets_columns=["left", "width", "top", "height"],
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

    # Make the combination of game_play, view, nfl_player_id and frame_time unique
    baseline_helmets = baseline_helmets[
        ~baseline_helmets.duplicated(
            subset=["game_play", "view", "nfl_player_id", "frame_time"]
        )
    ]

    baseline_helmets.drop(columns="datetime", inplace=True)

    labels = labels.astype({"nfl_player_id_1": "string", "nfl_player_id_2": "string"})
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
    labels["right_1"] = labels["left_1"] + labels["width_1"]
    labels["dx"] = labels["right_1"] - labels["left_2"]

    labels["bottom_1"] = labels["top_1"] + labels["height_1"]
    labels["dy"] = labels["bottom_1"] - labels["top_2"]

    has_intersection = (labels["dx"] > 0) & (labels["dy"] > 0)

    area_of_intersection = labels["dx"] * labels["dy"]
    area_of_intersection[~has_intersection] = 0

    area_of_union = (
        (labels["height_1"] * labels["width_1"])
        + (labels["height_2"] * labels["width_2"])
        - area_of_intersection
    )

    labels["iou"] = area_of_intersection / area_of_union

    labels.drop(columns=["right_1", "dx", "bottom_1", "dy"], inplace=True)

    return labels
