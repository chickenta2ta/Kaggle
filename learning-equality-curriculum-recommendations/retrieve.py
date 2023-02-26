import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from cuml.neighbors import NearestNeighbors
from essential_generators import DocumentGenerator
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

topics = topics[
    ["id", "title", "description", "category", "language", "parent", "has_content"]
]
topics = topics.astype(
    {
        "id": "S14",
        "title": "U",
        "description": "U",
        "category": "category",
        "language": "category",
        "parent": "S14",
        "has_content": "bool",
    }
)

content = content[["id", "title", "description", "text", "language"]]
content = content.astype(
    {
        "id": "S14",
        "title": "U",
        "description": "U",
        "text": "U",
        "language": "category",
    }
)


def remove_category_source(topics, correlations):
    correlations["topic_id"] = correlations["topic_id"].astype("S14")

    correlations = correlations.merge(
        topics, how="left", left_on=["topic_id"], right_on=["id"]
    )
    correlations = correlations[correlations["category"] != "source"]
    correlations = correlations[["topic_id", "content_ids"]]
    correlations.reset_index(drop=True, inplace=True)

    correlations["topic_id"] = correlations["topic_id"].str.decode("utf-8")

    topics = topics[topics["category"] != "source"]
    topics.reset_index(drop=True, inplace=True)

    return topics, correlations


def fillna_with_random_sentences(df, df_columns):
    gen = DocumentGenerator()
    for df_column in df_columns:
        is_null = df[df_column] == "nan"
        sentences = [gen.sentence() for _ in range(is_null.sum())]

        df.loc[is_null, df_column] = sentences

    return df


def retrieve(topics, content, path_to_weights):
    topics_columns = ["title", "description"]
    content_columns = ["title", "description", "text"]

    random.seed(42)
    topics = fillna_with_random_sentences(topics, topics_columns)
    content = fillna_with_random_sentences(content, content_columns)

    topics_with_content = topics[topics["has_content"]]
    topics_with_content.reset_index(drop=True, inplace=True)

    del topics
    gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(path_to_weights, device=device)

    correlations_pred = []
    for language, topics_language in topics_with_content.groupby(["language"]):
        content_language = content[content["language"] == language]

        if topics_language.empty or content_language.empty:
            continue

        topics_language.reset_index(drop=True, inplace=True)
        content_language.reset_index(drop=True, inplace=True)

        topics_content_distances_list = []
        topics_language_embeddings_dict = {}
        for content_column in content_columns:
            content_language_embeddings = model.encode(
                content_language[content_column].tolist(), batch_size=256, device=device
            )

            for topics_column in topics_columns:
                topics_language_embeddings = None

                if topics_column in topics_language_embeddings_dict:
                    topics_language_embeddings = topics_language_embeddings_dict[
                        topics_column
                    ]
                else:
                    topics_language_embeddings = model.encode(
                        topics_language[topics_column].tolist(),
                        batch_size=256,
                        device=device,
                    )
                    topics_language_embeddings_dict[
                        topics_column
                    ] = topics_language_embeddings

                neigh = NearestNeighbors(n_neighbors=100, metric="cosine")
                neigh.fit(content_language_embeddings)

                distances, content_indices = neigh.kneighbors(
                    topics_language_embeddings
                )

                del topics_language_embeddings
                gc.collect()

                distances = distances.flatten()
                content_indices = content_indices.flatten()

                distances = distances.reshape((-1, 1))
                content_indices = content_indices.reshape((-1, 1))

                content_distances = np.hstack((content_indices, distances))

                del distances, content_indices
                gc.collect()

                topics_indices = np.repeat(np.arange(len(topics_language)), 100)
                topics_indices = topics_indices.reshape((-1, 1))

                topics_content_distances = np.hstack(
                    (topics_indices, content_distances)
                )
                topics_content_distances = topics_content_distances.tolist()

                topics_content_distances_list.extend(topics_content_distances)

                del content_distances, topics_indices, topics_content_distances
                gc.collect()

            del content_language_embeddings
            gc.collect()

        topics_content_distances_list = pd.DataFrame(
            topics_content_distances_list,
            columns=["topic_index", "content_index", "distance"],
        )
        topics_content_distances_list = topics_content_distances_list.astype(
            {"topic_index": "int", "content_index": "int", "distance": "float"}
        )

        topics_content_distances_list = topics_content_distances_list.groupby(
            ["topic_index", "content_index"], as_index=False
        ).agg(count=("distance", "count"), distance=("distance", "mean"))
        topics_content_distances_list.sort_values(
            by=["topic_index", "count", "distance"],
            ascending=[True, False, True],
            inplace=True,
        )
        topics_content_distances_list = topics_content_distances_list.groupby(
            ["topic_index"]
        ).head(100)

        topics_content_distances = topics_content_distances_list.merge(
            topics_language["id"],
            how="left",
            left_on=["topic_index"],
            right_index=True,
        )
        topics_content_distances.rename(columns={"id": "topic_id"}, inplace=True)

        topics_content_distances = topics_content_distances.merge(
            content_language["id"],
            how="left",
            left_on=["content_index"],
            right_index=True,
        )
        topics_content_distances.rename(columns={"id": "content_id"}, inplace=True)

        topics_content_distances = topics_content_distances[
            ["topic_id", "content_id", "count", "distance"]
        ]
        topics_content_distances = topics_content_distances.astype(
            {
                "topic_id": "S14",
                "content_id": "S14",
                "count": "int",
                "distance": "float",
            }
        )

        correlations_pred.append(topics_content_distances)

        del topics_content_distances
        gc.collect()

    correlations_pred = pd.concat(correlations_pred)

    correlations_pred["topic_id"] = correlations_pred["topic_id"].str.decode("utf-8")
    correlations_pred["content_id"] = correlations_pred["content_id"].str.decode(
        "utf-8"
    )

    return correlations_pred


def calculate_recall(corr_true, corr_pred):
    corr_true["content_ids"] = corr_true["content_ids"].str.split()
    corr_true = corr_true.explode("content_ids")

    corr_true = corr_true.merge(
        corr_pred,
        how="left",
        left_on=["topic_id", "content_ids"],
        right_on=["topic_id", "content_id"],
    )

    rowwise_len = len(corr_true)
    print(
        f"overall recall: {corr_true['content_id'].notnull().sum() / rowwise_len:.2%}"
    )

    corr_true = corr_true.groupby(["topic_id"]).agg(
        {"content_ids": "count", "content_id": "count"}
    )
    corr_true["recall"] = corr_true["content_id"] / corr_true["content_ids"]

    topicwise_len = len(corr_true)
    print(f"recall of 100%: {(corr_true['recall'] == 1).sum() / topicwise_len:.2%}")
    print(f"recall of 80%: {(corr_true['recall'] >= 0.8).sum() / topicwise_len:.2%}")
    print(f"recall of 50%: {(corr_true['recall'] >= 0.5).sum() / topicwise_len:.2%}")
    print(f"recall of 0%: {(corr_true['recall'] == 0).sum() / topicwise_len:.2%}")


topics, correlations = remove_category_source(topics, correlations)
correlations_pred = retrieve(topics, content, PATH_TO_WEIGHTS)
calculate_recall(correlations, correlations_pred)
