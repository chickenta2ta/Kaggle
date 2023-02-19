import os

import numpy as np
import pandas as pd
import torch
from cuml.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

PATH_TO_INPUT = "../input/learning-equality-curriculum-recommendations"
PATH_TO_OUTPUT = "."

TOPICS = "topics.csv"
CONTENT = "content.csv"
CORRELATIONS = "correlations.csv"

PATH_TO_WEIGHTS = "../input/all-minilm-l6-v2"

topics = pd.read_csv(os.path.join(PATH_TO_INPUT, TOPICS))
content = pd.read_csv(os.path.join(PATH_TO_INPUT, CONTENT))
correlations = pd.read_csv(os.path.join(PATH_TO_INPUT, CORRELATIONS))


def retrieve(topics, content, path_to_weights):
    topics_with_content = topics[topics["has_content"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(path_to_weights, device=device)

    correlations_preds = []
    for language, topics_language in topics_with_content.groupby(["language"]):
        content_language = content[content["language"] == language]

        topics_language_embeddings = model.encode(
            topics_language["title"].tolist(), device=device
        )
        content_language_embeddings = model.encode(
            content_language["title"].tolist(), device=device
        )

        neigh = NearestNeighbors(n_neighbors=100, metric="cosine")
        neigh.fit(content_language_embeddings)

        indices = neigh.kneighbors(topics_language_embeddings, return_distance=False)
        content_language_id = content_language["id"]
        content_language_id = np.broadcast_to(
            content_language_id, (len(topics_language), content_language_id.size)
        )
        content_language_id = np.take_along_axis(content_language_id, indices, axis=1)

        topics_language_id = topics_language["id"].to_numpy()
        topics_language_id = topics_language_id.reshape((-1, 1))

        correlations_pred = np.hstack((topics_language_id, content_language_id))
        correlations_pred = correlations_pred.tolist()

        correlations_preds.extend(correlations_pred)

    columns = ["topic_id"]
    columns += [f"content_id_{i + 1}" for i in range(100)]
    correlations_pred = pd.DataFrame(correlations_preds, columns=columns)

    correlations_pred = correlations_pred.melt(
        id_vars=["topic_id"], value_name="content_id"
    )
    correlations_pred = correlations_pred[["topic_id", "content_id"]]

    return correlations_pred


def calculate_recall(correlations_true, correlations_pred):
    correlations_true["content_ids"] = correlations_true["content_ids"].str.split()
    correlations_true = correlations_true.explode("content_ids")

    correlations_true = correlations_true.merge(
        correlations_pred,
        how="left",
        left_on=["topic_id", "content_ids"],
        right_on=["topic_id", "content_id"],
    )
    correlations_true = correlations_true.groupby(["topic_id"]).agg(
        {"content_ids": "count", "content_id": "count"}
    )
    correlations_true["recall"] = (
        correlations_true["content_id"] / correlations_true["content_ids"]
    )

    len_ = len(correlations_true)
    print(f"recall of 100%: {(correlations_true['recall'] == 1).sum() / len_:.2%}")
    print(f"recall of 80%: {(correlations_true['recall'] >= 0.8).sum() / len_:.2%}")
    print(f"recall of 50%: {(correlations_true['recall'] >= 0.5).sum() / len_:.2%}")
    print(f"recall of 0%: {(correlations_true['recall'] == 0).sum() / len_:.2%}")


correlations_pred = retrieve(topics, content, PATH_TO_WEIGHTS)
calculate_recall(correlations, correlations_pred)
