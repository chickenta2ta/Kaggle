import gc
import os

import cv2
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from scipy.optimize import minimize
from skimage import io
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision.models import resnet152
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

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

PATH_TO_TEST_VIDEOS = "../input/nfl-player-contact-detection/test"

PATH_TO_TRAIN_FRAMES = "../input/nfl-player-contact-detection-frames"
PATH_TO_TEST_FRAMES = "../../tmp/nfl-player-contact-detection-frames"

PATH_TO_WEIGHTS = "../input/nfl-player-contact-detection-weights"

RESNET152 = "resnet152.pth"
RESNET152_PLAYER_SIDELINE = "resnet152-player-sideline.pth"
RESNET152_PLAYER_ENDZONE = "resnet152-player-endzone.pth"
RESNET152_GROUND_SIDELINE = "resnet152-ground-sideline.pth"
RESNET152_GROUND_ENDZONE = "resnet152-ground-endzone.pth"
KEYPOINTRCNN_RESNET50_FPN = "keypointrcnn-resnet50-fpn.pth"

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

resnet152_weights = os.path.join(PATH_TO_WEIGHTS, RESNET152)
resnet152_player_sideline_weights = os.path.join(
    PATH_TO_WEIGHTS, RESNET152_PLAYER_SIDELINE
)
resnet152_player_endzone_weights = os.path.join(
    PATH_TO_WEIGHTS, RESNET152_PLAYER_ENDZONE
)
resnet152_ground_sideline_weights = os.path.join(
    PATH_TO_WEIGHTS, RESNET152_GROUND_SIDELINE
)
resnet152_ground_endzone_weights = os.path.join(
    PATH_TO_WEIGHTS, RESNET152_GROUND_ENDZONE
)
keypointrcnn_resnet50_fpn_weights = os.path.join(
    PATH_TO_WEIGHTS, KEYPOINTRCNN_RESNET50_FPN
)

# Endzone2 should be ignored as it is a merging error or something
train_baseline_helmets = train_baseline_helmets[
    train_baseline_helmets["view"] != "Endzone2"
]


def video_to_frames(video_metadata, path_to_videos, path_to_frames):
    os.makedirs(path_to_frames, exist_ok=True)
    for (game_play, view), _ in video_metadata.groupby(["game_play", "view"]):
        game_play_view = game_play + "_" + view
        cap = cv2.VideoCapture(os.path.join(path_to_videos, game_play_view + ".mp4"))

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imwrite(
                os.path.join(path_to_frames, game_play_view + f"_{i}.jpg"), frame
            )
            i += 1

        cap.release()


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

    frame_time_to_frame = baseline_helmets.groupby(
        ["game_play", "view", "frame_time"], as_index=False
    ).agg({"frame": "median"})
    frame_time_to_frame["frame"] = frame_time_to_frame["frame"].round()
    baseline_helmets.drop(columns="frame", inplace=True)
    baseline_helmets = baseline_helmets.merge(
        frame_time_to_frame, how="left", on=["game_play", "view", "frame_time"]
    )

    baseline_helmets = baseline_helmets.groupby(
        ["game_play", "view", "nfl_player_id", "frame_time", "frame"], as_index=False
    ).agg({column_name: "mean" for column_name in baseline_helmets_columns})

    labels["datetime"] = pd.to_datetime(labels["datetime"])
    labels["datetime"] = labels["datetime"].dt.round("100L")

    baseline_helmets = baseline_helmets.astype({"nfl_player_id": "string"})

    for view in ["Sideline", "Endzone"]:
        view_lower = view.lower()

        baseline_helmets_view = baseline_helmets[
            baseline_helmets["view"] == view
        ].copy()

        labels = labels.merge(
            baseline_helmets_view[
                ["game_play", "frame_time", "frame"]
            ].drop_duplicates(),
            how="left",
            left_on=["game_play", "datetime"],
            right_on=["game_play", "frame_time"],
        )
        labels.rename(columns={"frame": f"frame_{view_lower}"}, inplace=True)

        labels.drop(columns="frame_time", inplace=True)
        baseline_helmets_view.drop(columns="frame", inplace=True)

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
        "team",
        "position",
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


def append_2(column_names):
    appended_column_names = []
    for column_name in column_names:
        appended_column_names.append(column_name + "_2")

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


def get_indices_of_closer_than_threshold(labels, threshold=2):
    necessary_columns = ["distance"]
    if column_exists(labels, necessary_columns):
        is_close = (labels["distance"] <= threshold) | labels["distance"].isnull()
        return is_close
    else:
        return pd.Series(True, index=labels.index)


def calculate_iou(labels):
    feature_columns = []
    for view in ["Sideline", "Endzone"]:
        view_lower = view.lower()

        labels[f"right_{view_lower}_1"] = (
            labels[f"left_{view_lower}_1"] + labels[f"width_{view_lower}_1"]
        )
        labels[f"right_{view_lower}_2"] = (
            labels[f"left_{view_lower}_2"] + labels[f"width_{view_lower}_2"]
        )
        labels[f"dx_{view_lower}"] = labels[
            [f"right_{view_lower}_1", f"right_{view_lower}_2"]
        ].min(axis=1, skipna=False) - labels[
            [f"left_{view_lower}_1", f"left_{view_lower}_2"]
        ].max(
            axis=1, skipna=False
        )

        labels[f"bottom_{view_lower}_1"] = (
            labels[f"top_{view_lower}_1"] + labels[f"height_{view_lower}_1"]
        )
        labels[f"bottom_{view_lower}_2"] = (
            labels[f"top_{view_lower}_2"] + labels[f"height_{view_lower}_2"]
        )
        labels[f"dy_{view_lower}"] = labels[
            [f"bottom_{view_lower}_1", f"bottom_{view_lower}_2"]
        ].min(axis=1, skipna=False) - labels[
            [f"top_{view_lower}_1", f"top_{view_lower}_2"]
        ].max(
            axis=1, skipna=False
        )

        has_intersection = (labels[f"dx_{view_lower}"] > 0) & (
            labels[f"dy_{view_lower}"] > 0
        )

        area_of_intersection = labels[f"dx_{view_lower}"] * labels[f"dy_{view_lower}"]
        area_of_intersection[~has_intersection] = 0

        area_of_union = (
            (labels[f"height_{view_lower}_1"] * labels[f"width_{view_lower}_1"])
            + (labels[f"height_{view_lower}_2"] * labels[f"width_{view_lower}_2"])
            - area_of_intersection
        )

        labels[f"iou_{view_lower}"] = area_of_intersection / area_of_union
        labels.fillna(value={f"iou_{view_lower}": 0})

        labels.drop(
            columns=[
                f"right_{view_lower}_1",
                f"right_{view_lower}_2",
                f"dx_{view_lower}",
                f"bottom_{view_lower}_1",
                f"bottom_{view_lower}_2",
                f"dy_{view_lower}",
            ],
            inplace=True,
        )

        feature_columns.append(f"iou_{view_lower}")

    return labels, feature_columns


def calculate_helmets_distance(labels):
    feature_columns = []
    for view in ["Sideline", "Endzone"]:
        view_lower = view.lower()

        width_1, height_1 = (
            labels[f"width_{view_lower}_1"],
            labels[f"height_{view_lower}_1"],
        )
        width_2, height_2 = (
            labels[f"width_{view_lower}_2"],
            labels[f"height_{view_lower}_2"],
        )

        center_x_1 = labels[f"left_{view_lower}_1"] + (width_1 / 2)
        center_y_1 = labels[f"top_{view_lower}_1"] + (height_1 / 2)

        center_x_2 = labels[f"left_{view_lower}_2"] + (width_2 / 2)
        center_y_2 = labels[f"top_{view_lower}_2"] + (height_2 / 2)

        labels[f"helmets_distance_{view_lower}"] = np.sqrt(
            ((center_x_1 - center_x_2) ** 2) + ((center_y_1 - center_y_2) ** 2)
        ) / (((width_1 * height_1) + (width_2 * height_2)) / 2)
        feature_columns.append(f"helmets_distance_{view_lower}")

    return labels, feature_columns


def calculate_moving_average(labels, column_names, windows=[10]):
    feature_columns = []

    labels.sort_values(
        by=["game_play", "nfl_player_id_1", "nfl_player_id_2", "step"], inplace=True
    )

    for column_name in column_names:
        for window in windows:
            new_column_name = column_name + f"_moving_average_{window}"

            labels[new_column_name] = (
                labels.groupby(
                    ["game_play", "nfl_player_id_1", "nfl_player_id_2"],
                    as_index=False,
                    sort=False,
                )[column_name]
                .rolling(window)
                .mean()[column_name]
            )
            # Not filling NaN performed better
            # labels[new_column_name].fillna(labels[column_name], inplace=True)

            feature_columns.append(new_column_name)

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
    sum_columns=["sa", "iou", "helmets_distance"],
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

    necessary_columns = ["left", "width", "top", "height"]
    necessary_columns = append_sideline_and_endzone(necessary_columns)
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_helmets_distance(labels)
        feature_columns += columns

    necessary_columns = ["team"]
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels["is_same_team"] = (labels["team_1"] == labels["team_2"]).astype("int")
        feature_columns.append("is_same_team")

    necessary_columns = ["position"]
    necessary_columns = append_1_and_2(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels["is_same_position"] = (
            labels["position_1"] == labels["position_2"]
        ).astype("int")
        feature_columns.append("is_same_position")

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


def pose_estimation(
    labels,
    path_to_frames,
    path_to_weights,
):
    feature_columns = []

    model = keypointrcnn_resnet50_fpn(pretrained_backbone=False)
    model.load_state_dict(torch.load(path_to_weights))
    torchvision.models.detection._utils.overwrite_eps(model, 0.0)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    for view in ["Sideline", "Endzone"]:
        view_lower = view.lower()

        necessary_columns = ["left", "width", "top", "height"]
        new_column_names = [
            "nose_y_relative_to_ankle_y",
            "left_wrist_y_relative_to_ankle_y",
            "right_wrist_y_relative_to_ankle_y",
            "left_knee_y_relative_to_ankle_y",
            "right_knee_y_relative_to_ankle_y",
            "min_y_relative_to_ankle_y",
        ]

        if view == "Sideline":
            necessary_columns = append_sideline(necessary_columns)
            new_column_names = append_sideline(new_column_names)
        else:
            necessary_columns = append_endzone(necessary_columns)
            new_column_names = append_endzone(new_column_names)
        necessary_columns = append_1(necessary_columns)
        new_column_names = append_1(new_column_names)

        for (game_play, frame), group in labels.groupby(
            ["game_play", f"frame_{view_lower}"]
        ):
            if np.isnan(frame):
                continue

            frame = int(frame)
            img_name = os.path.join(path_to_frames, f"{game_play}_{view}_{frame}.jpg")

            image = io.imread(img_name)
            image = torchvision.transforms.functional.to_tensor(image)
            image = image.unsqueeze(0)

            image = image.to(device)

            prediction = model(image)[0]

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            keypoints = prediction["keypoints"]

            boxes = boxes.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            keypoints = keypoints.detach().cpu().numpy()

            is_confident = scores > 0.25
            boxes = boxes[is_confident]
            keypoints = keypoints[is_confident]

            boxes_topleft, boxes_bottomright = np.hsplit(boxes, 2)

            for row in group[necessary_columns].itertuples():
                index, left, width, top, height = row

                if np.isnan(left):
                    continue

                right = left + width
                bottom = top + height

                left += width * 0.05
                top += height * 0.05
                right -= width * 0.05
                bottom -= height * 0.05

                boxes_topleft_player = boxes_topleft <= np.array([left, top])
                boxes_bottomright_player = boxes_bottomright >= np.array(
                    [right, bottom]
                )
                boxes_player = np.hstack(
                    (boxes_topleft_player, boxes_bottomright_player)
                )
                boxes_player = np.all(boxes_player, axis=1)

                keypoints_player = keypoints[boxes_player]

                if keypoints_player.size == 0:
                    continue

                keypoint_player = keypoints_player[0]
                (
                    nose_y,
                    left_wrist_y,
                    right_wrist_y,
                    left_knee_y,
                    right_knee_y,
                    left_ankle_y,
                    right_ankle_y,
                ) = keypoint_player[[0, 9, 10, 13, 14, 15, 16], 1]

                ankle_y = (left_ankle_y + right_ankle_y) / 2

                box_size = boxes[boxes_player][0].size

                nose_y_relative_to_ankle_y = (ankle_y - nose_y) / box_size
                left_wrist_y_relative_to_ankle_y = (ankle_y - left_wrist_y) / box_size
                right_wrist_y_relative_to_ankle_y = (ankle_y - right_wrist_y) / box_size
                left_knee_y_relative_to_ankle_y = (ankle_y - left_knee_y) / box_size
                right_knee_y_relative_to_ankle_y = (ankle_y - right_knee_y) / box_size

                min_y_relative_to_ankle_y = (
                    ankle_y
                    - max(
                        nose_y, left_wrist_y, right_wrist_y, left_knee_y, right_knee_y
                    )
                ) / box_size

                labels.loc[index, new_column_names] = (
                    nose_y_relative_to_ankle_y,
                    left_wrist_y_relative_to_ankle_y,
                    right_wrist_y_relative_to_ankle_y,
                    left_knee_y_relative_to_ankle_y,
                    right_knee_y_relative_to_ankle_y,
                    min_y_relative_to_ankle_y,
                )

        feature_columns += new_column_names

    labels["min_y_relative_to_ankle_y_1"] = labels[
        ["min_y_relative_to_ankle_y_sideline_1", "min_y_relative_to_ankle_y_endzone_1"]
    ].mean(axis=1)
    feature_columns.append("min_y_relative_to_ankle_y_1")

    return labels, feature_columns


def create_features_for_ground(
    labels,
    path_to_frames,
    path_to_weights,
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

    necessary_columns = ["left", "width", "top", "height"]
    necessary_columns = append_sideline_and_endzone(necessary_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = pose_estimation(labels, path_to_frames, path_to_weights)
        feature_columns += columns

    # Add moving average features
    necessary_columns = append_1(append_1_moving_average_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(
            labels, necessary_columns, windows=[10, 20, 30]
        )
        feature_columns += columns

    necessary_columns = append_sideline(append_sideline_1_moving_average_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(
            labels, necessary_columns, windows=[10, 20, 30]
        )
        feature_columns += columns

    necessary_columns = append_endzone(append_endzone_1_moving_average_columns)
    necessary_columns = append_1(necessary_columns)
    if column_exists(labels, necessary_columns):
        labels, columns = calculate_moving_average(
            labels, necessary_columns, windows=[10, 20, 30]
        )
        feature_columns += columns

    return labels, feature_columns


def create_features_for_ground_stub():
    labels = pd.read_csv(
        os.path.join(
            "../input/nfl-player-contact-detection-preprocessed-data",
            "train_labels_ground.csv",
        )
    )
    feature_columns = [
        "x_position_1",
        "y_position_1",
        "speed_1",
        "distance_1",
        "direction_1",
        "orientation_1",
        "acceleration_1",
        "sa_1",
        "left_sideline_1",
        "width_sideline_1",
        "top_sideline_1",
        "height_sideline_1",
        "width_endzone_1",
        "top_endzone_1",
        "nose_y_relative_to_ankle_y_sideline_1",
        "left_wrist_y_relative_to_ankle_y_sideline_1",
        "right_wrist_y_relative_to_ankle_y_sideline_1",
        "left_knee_y_relative_to_ankle_y_sideline_1",
        "right_knee_y_relative_to_ankle_y_sideline_1",
        "min_y_relative_to_ankle_y_sideline_1",
        "nose_y_relative_to_ankle_y_endzone_1",
        "left_wrist_y_relative_to_ankle_y_endzone_1",
        "right_wrist_y_relative_to_ankle_y_endzone_1",
        "left_knee_y_relative_to_ankle_y_endzone_1",
        "right_knee_y_relative_to_ankle_y_endzone_1",
        "min_y_relative_to_ankle_y_endzone_1",
        "min_y_relative_to_ankle_y_1",
        "x_position_1_moving_average_10",
        "x_position_1_moving_average_20",
        "x_position_1_moving_average_30",
        "y_position_1_moving_average_10",
        "y_position_1_moving_average_20",
        "y_position_1_moving_average_30",
        "speed_1_moving_average_10",
        "speed_1_moving_average_20",
        "speed_1_moving_average_30",
        "orientation_1_moving_average_10",
        "orientation_1_moving_average_20",
        "orientation_1_moving_average_30",
        "sa_1_moving_average_10",
        "sa_1_moving_average_20",
        "sa_1_moving_average_30",
        "left_sideline_1_moving_average_10",
        "left_sideline_1_moving_average_20",
        "left_sideline_1_moving_average_30",
        "top_sideline_1_moving_average_10",
        "top_sideline_1_moving_average_20",
        "top_sideline_1_moving_average_30",
        "height_sideline_1_moving_average_10",
        "height_sideline_1_moving_average_20",
        "height_sideline_1_moving_average_30",
        "width_endzone_1_moving_average_10",
        "width_endzone_1_moving_average_20",
        "width_endzone_1_moving_average_30",
        "top_endzone_1_moving_average_10",
        "top_endzone_1_moving_average_20",
        "top_endzone_1_moving_average_30",
    ]

    return labels, feature_columns


def train_lightgbm(
    X_train,
    y_train,
    groups,
    param={
        "objective": "binary",
        "num_boost_round": 1_000,
        "learning_rate": 0.03,
        "device_type": "gpu" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "force_row_wise": True,
        "early_stopping_round": 100,
        "metric": "auc",
    },
):
    group_kfold = GroupKFold()

    # This weird implementation is to avoid LightGBM warnings
    num_boost_round = param["num_boost_round"]
    stopping_rounds = param["early_stopping_round"]

    param_copy = param.copy()
    del param_copy["num_boost_round"], param_copy["early_stopping_round"]

    models = []
    for i, (train_index, test_index) in enumerate(
        group_kfold.split(X_train, y_train, groups)
    ):
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

        models.append(model)

    return models


def get_necessary_columns_for_resnet():
    necessary_columns = ["game_play", "contact"]

    columns = ["frame"]
    columns = append_sideline_and_endzone(columns)
    necessary_columns += columns

    columns = ["left", "width", "top", "height"]
    columns = append_sideline_and_endzone(columns)
    columns = append_1_and_2(columns)
    necessary_columns += columns

    return necessary_columns


def remove_nan_rows(labels, is_player, view_lower):
    labels = labels[~labels[f"frame_{view_lower}"].isnull()]
    labels = labels[~labels[f"left_{view_lower}_1"].isnull()]
    if is_player:
        labels = labels[~labels[f"left_{view_lower}_2"].isnull()]

    return labels


def preprocess_labels_for_resnet(labels, is_player, view_lower):
    is_close = get_indices_of_closer_than_threshold(labels)
    necessary_columns = get_necessary_columns_for_resnet()

    labels_copy = labels.loc[is_close, necessary_columns].copy()
    labels_copy = remove_nan_rows(labels_copy, is_player, view_lower)

    return labels_copy


class CenterCrop(object):
    def __init__(self, view):
        self.view = view

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        columns = ["left", "width", "top", "height"]

        if self.view == "Sideline":
            columns = append_sideline(columns)
        else:
            columns = append_endzone(columns)
        columns = append_1(columns) + append_2(columns)

        bounding_boxes = label[columns]
        bounding_boxes = bounding_boxes.to_numpy()
        bounding_boxes = np.reshape(bounding_boxes, (-1, 4))

        image = self.draw_rectangle(image, bounding_boxes)

        center_x, center_y = self.get_center(bounding_boxes)
        image = image[center_y - 128 : center_y + 128, center_x - 128 : center_x + 128]

        return image

    def draw_rectangle(self, image, rectangles):
        for rectangle in rectangles:
            if np.isnan(rectangle[0]):
                continue

            left, width, top, height = map(int, rectangle)
            image = cv2.rectangle(
                image, (left, top), (left + width, top + height), (255, 0, 0)
            )

        return image

    def get_center(self, bounding_boxes):
        m = np.nanmean(bounding_boxes, axis=0)

        left, width, top, height = m[0], m[1], m[2], m[3]

        center_x = left + (width / 2)
        center_y = top + (height / 2)

        center_x = int(center_x)
        center_y = int(center_y)

        center_x = np.clip(center_x, 128, 1151)
        center_y = np.clip(center_y, 128, 591)

        return center_x, center_y


class NFLDataset(Dataset):
    def __init__(
        self,
        labels,
        path_to_frames,
        view,
    ):
        self.labels = labels
        self.path_to_frames = path_to_frames
        self.view = view
        self.transform = Compose(
            [
                CenterCrop(view),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = Compose(
            [
                Lambda(lambda y: torch.tensor(y, dtype=torch.float32)),
                Lambda(lambda y: torch.unsqueeze(y, 0)),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        label = self.labels.iloc[idx]

        view_lower = self.view.lower()
        game_play, frame = label[["game_play", f"frame_{view_lower}"]]
        frame = int(frame)
        img_name = os.path.join(
            self.path_to_frames, f"{game_play}_{self.view}_{frame}.jpg"
        )

        image = io.imread(img_name)
        image = self.transform({"image": image, "label": label})

        contact = label["contact"]
        contact = self.target_transform(contact)

        return image, contact


def _predict_resnet(model, labels, path_to_frames, view):
    data = NFLDataset(labels, path_to_frames, view)
    dataloader = torch.utils.data.DataLoader(data, batch_size=512)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)

            y = model(x)
            y = torch.sigmoid(y)

            y = torch.flatten(y)
            y = y.tolist()
            probabilities += y

    return probabilities


def matthews_corrcoef_(x, y_train, y_pred_train):
    mcc = matthews_corrcoef(y_train, y_pred_train > x[0])
    return -mcc


def optimize_threshold(y_true, y_pred, x0=[0.5]):
    res = minimize(matthews_corrcoef_, x0, args=(y_true, y_pred), method="Nelder-Mead")

    y_pred = (y_pred > res.x[0]).astype("int")
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc, res


def train_resnet(
    train_labels,
    test_labels,
    path_to_frames,
    path_to_weights,
    is_player,
    view,
):
    view_lower = view.lower()

    train_labels_copy = preprocess_labels_for_resnet(
        train_labels, is_player, view_lower
    )

    torch.manual_seed(0)

    training_data = NFLDataset(train_labels_copy, path_to_frames, view)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=32)

    model = resnet152()
    model.load_state_dict(torch.load(path_to_weights))
    model.fc = nn.Linear(2048, 1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    model_name = f"resnet152-{'player' if is_player else 'ground'}-{view_lower}.pth"
    torch.save(model.state_dict(), os.path.join(".", model_name))

    test_labels_copy = preprocess_labels_for_resnet(test_labels, is_player, view_lower)

    y_true = test_labels_copy["contact"]
    y_pred = _predict_resnet(model, test_labels_copy, path_to_frames, view)

    mcc, res = optimize_threshold(y_true, y_pred)

    model_description = f"ResNet for {'Player' if is_player else 'Ground'}/{view}"
    print(f"MCC ({model_description}): {mcc}")
    print(f"OptimizeResult ({model_description}): {res.x[0]}")


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


def predict_lightgbm(
    models,
    X,
):
    y = np.zeros(len(X))

    for model in models:
        y += model.predict(X, num_iteration=model.best_iteration) / len(models)

    return y


def predict_resnet(
    labels,
    path_to_frames,
    path_to_weights,
    is_player,
    view,
):
    view_lower = view.lower()

    labels_copy = preprocess_labels_for_resnet(labels, is_player, view_lower)

    model = resnet152()
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(path_to_weights))

    probabilities = _predict_resnet(model, labels_copy, path_to_frames, view)

    column_name = "probability_of_contact_resnet"
    if is_player:
        column_name += f"_player_{view_lower}"
    else:
        column_name += f"_ground_{view_lower}"

    labels_copy[column_name] = probabilities
    labels_copy = labels_copy[[column_name]]

    return labels_copy


def predict_all(
    models_lightgbm_player,
    models_lightgbm_ground,
    labels,
    feature_columns_player,
    feature_columns_ground,
    path_to_frames,
    weights_resnet152_player_sideline,
    weights_resnet152_player_endzone,
    weights_resnet152_ground_sideline,
    weights_resnet152_ground_endzone,
    weights_keypointrcnn_resnet50_fpn,
    prints_mcc=False,
):
    labels_player, labels_ground = separate_player_and_ground(labels)

    # Do not reset index. We rather want to keep the order of sample_submission.csv
    # test_labels_player.reset_index(drop=True, inplace=True)
    # test_labels_ground.reset_index(drop=True, inplace=True)

    labels_player, _ = create_features_for_player(labels_player, is_training=False)
    labels_ground, _ = create_features_for_ground(
        labels_ground, path_to_frames, weights_keypointrcnn_resnet50_fpn
    )

    labels_player["probability_of_contact_lightgbm_player"] = predict_lightgbm(
        models_lightgbm_player, labels_player[feature_columns_player]
    )

    probability_of_contact_resnet_player_sideline = predict_resnet(
        labels_player,
        path_to_frames,
        weights_resnet152_player_sideline,
        True,
        "Sideline",
    )
    labels_player = labels_player.merge(
        probability_of_contact_resnet_player_sideline,
        how="left",
        left_index=True,
        right_index=True,
    )

    probability_of_contact_resnet_player_endzone = predict_resnet(
        labels_player,
        path_to_frames,
        weights_resnet152_player_endzone,
        True,
        "Endzone",
    )
    labels_player = labels_player.merge(
        probability_of_contact_resnet_player_endzone,
        how="left",
        left_index=True,
        right_index=True,
    )

    labels_player["probability_of_contact_resnet_player"] = labels_player[
        [
            "probability_of_contact_resnet_player_sideline",
            "probability_of_contact_resnet_player_endzone",
        ]
    ].mean(axis=1)

    probability_of_contact_player = labels_player[
        [
            "probability_of_contact_lightgbm_player",
            "probability_of_contact_resnet_player",
        ]
    ]
    mx = ma.masked_array(
        probability_of_contact_player, mask=probability_of_contact_player.isnull()
    )
    average = ma.average(mx, axis=1, weights=[0.6, 0.4])
    labels_player["probability_of_contact"] = average.filled(fill_value=np.nan)

    # probability_of_contact_[player/ground] is used when calculating mcc
    labels_player["probability_of_contact_player"] = labels_player[
        "probability_of_contact"
    ]

    labels_player["contact"] = (
        labels_player["probability_of_contact"] > 0.4164
    ).astype("int")

    labels_ground["probability_of_contact_lightgbm_ground"] = predict_lightgbm(
        models_lightgbm_ground, labels_ground[feature_columns_ground]
    )

    probability_of_contact_resnet_ground_sideline = predict_resnet(
        labels_ground,
        path_to_frames,
        weights_resnet152_ground_sideline,
        False,
        "Sideline",
    )
    labels_ground = labels_ground.merge(
        probability_of_contact_resnet_ground_sideline,
        how="left",
        left_index=True,
        right_index=True,
    )

    probability_of_contact_resnet_ground_endzone = predict_resnet(
        labels_ground,
        path_to_frames,
        weights_resnet152_ground_endzone,
        False,
        "Endzone",
    )
    labels_ground = labels_ground.merge(
        probability_of_contact_resnet_ground_endzone,
        how="left",
        left_index=True,
        right_index=True,
    )

    labels_ground["probability_of_contact_resnet_ground"] = labels_ground[
        [
            "probability_of_contact_resnet_ground_sideline",
            "probability_of_contact_resnet_ground_endzone",
        ]
    ].mean(axis=1)

    probability_of_contact_ground = labels_ground[
        [
            "probability_of_contact_lightgbm_ground",
            "probability_of_contact_resnet_ground",
        ]
    ]
    mx = ma.masked_array(
        probability_of_contact_ground, mask=probability_of_contact_ground.isnull()
    )
    average = ma.average(mx, axis=1, weights=[0.5, 0.5])
    labels_ground["probability_of_contact"] = average.filled(fill_value=np.nan)

    # probability_of_contact_[player/ground] is used when calculating mcc
    labels_ground["probability_of_contact_ground"] = labels_ground[
        "probability_of_contact"
    ]

    labels_ground["contact"] = (
        labels_ground["probability_of_contact"] > 0.2500
    ).astype("int")

    labels = pd.concat([labels_player, labels_ground])
    labels.sort_index(inplace=True)

    is_close = get_indices_of_closer_than_threshold(labels)

    if prints_mcc:
        probability_columns = [
            "probability_of_contact_player",
            "probability_of_contact_lightgbm_player",
            "probability_of_contact_resnet_player",
            "probability_of_contact_resnet_player_sideline",
            "probability_of_contact_resnet_player_endzone",
            "probability_of_contact_ground",
            "probability_of_contact_lightgbm_ground",
            "probability_of_contact_resnet_ground",
            "probability_of_contact_resnet_ground_sideline",
            "probability_of_contact_resnet_ground_endzone",
        ]

        for column_name in probability_columns:
            labels.loc[~is_close, column_name] = 0

            is_null = labels[column_name].isnull()

            y_true = labels.loc[~is_null, "contact"]
            y_pred = labels.loc[~is_null, column_name]

            mcc, res = optimize_threshold(y_true, y_pred)

            print(f"MCC ({column_name}): {mcc}")
            print(f"OptimizeResult ({column_name}): {res.x[0]}")

    labels.loc[~is_close, "contact"] = 0

    labels[["contact_id", "contact"]].to_csv(
        os.path.join(PATH_TO_OUTPUT, SUBMISSION), index=False
    )


video_to_frames(test_video_metadata, PATH_TO_TEST_VIDEOS, PATH_TO_TEST_FRAMES)
gc.collect()

train_labels = join_player_tracking_and_baseline_helmets_to_labels(
    train_labels, train_player_tracking, train_baseline_helmets, train_video_metadata
)

del train_player_tracking, train_baseline_helmets, train_video_metadata
gc.collect()

gss = GroupShuffleSplit(n_splits=1, random_state=42)
train_index, test_index = next(
    gss.split(train_labels, groups=train_labels["game_play"])
)
train_labels, test_labels = train_labels.loc[train_index], train_labels.loc[test_index]
train_labels.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)

del train_index, test_index
gc.collect()

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
# train_labels_ground, feature_columns_ground = create_features_for_ground(
#     train_labels_ground, PATH_TO_TRAIN_FRAMES, keypointrcnn_resnet50_fpn_weights
# )

# Use stub to avoid long computation time
train_labels_ground, feature_columns_ground = create_features_for_ground_stub()

models_lightgbm_player = train_lightgbm(
    train_labels_player[feature_columns_player],
    train_labels_player["contact"],
    train_labels_player["game_play"],
)
models_lightgbm_ground = train_lightgbm(
    train_labels_ground[feature_columns_ground],
    train_labels_ground["contact"],
    train_labels_ground["game_play"],
)

# test_labels_player, _ = separate_player_and_ground(test_labels)
# train_resnet(
#     train_labels_player,
#     test_labels_player,
#     PATH_TO_TRAIN_FRAMES,
#     resnet152_weights,
#     True,
#     "Sideline",
# )

del train_labels_player, train_labels_ground
gc.collect()

sample_submission = split_contact_id(sample_submission)
sample_submission = join_datetime_to_labels(sample_submission, test_player_tracking)

sample_submission = join_player_tracking_and_baseline_helmets_to_labels(
    sample_submission, test_player_tracking, test_baseline_helmets, test_video_metadata
)

del test_player_tracking, test_baseline_helmets, test_video_metadata
gc.collect()

predict_all(
    models_lightgbm_player,
    models_lightgbm_ground,
    sample_submission,
    feature_columns_player,
    feature_columns_ground,
    PATH_TO_TEST_FRAMES,
    resnet152_player_sideline_weights,
    resnet152_player_endzone_weights,
    resnet152_ground_sideline_weights,
    resnet152_ground_endzone_weights,
    keypointrcnn_resnet50_fpn_weights,
)

# Evaluate models before submission
predict_all(
    models_lightgbm_player,
    models_lightgbm_ground,
    test_labels,
    feature_columns_player,
    feature_columns_ground,
    PATH_TO_TRAIN_FRAMES,
    resnet152_player_sideline_weights,
    resnet152_player_endzone_weights,
    resnet152_ground_sideline_weights,
    resnet152_ground_endzone_weights,
    keypointrcnn_resnet50_fpn_weights,
    prints_mcc=True,
)

# Visualize feature importance
fig, axs = plt.subplots(2, figsize=(6.4 * 5, 4.8 * 5))
lgb.plot_importance(
    models_lightgbm_player[0],
    ax=axs[0],
    title="Feature importance of player",
)
lgb.plot_importance(
    models_lightgbm_ground[0],
    ax=axs[1],
    title="Feature importance of ground",
)
plt.show()
