from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from configurations.data_frame_columns import SPLIT_ARTICLES, SPLIT_ARTICLES_EMBEDDING, PARTICIPANT_ID, BINARY_LABELS
from configurations.model_configurations import SENTENCE_TRANSFORMER_MODEL, BATCH_SIZE, SHOW_PROGRESS_BAR, \
    NORMALIZE_EMBEDDINGS
from typing import Callable


def get_embeddings(documents: List) -> np.ndarray:
    device = ''.join('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    sentence_transformer = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    print(sentence_transformer.max_seq_length)
    document_embeddings = sentence_transformer.encode(documents, batch_size=BATCH_SIZE,
                                                      show_progress_bar=SHOW_PROGRESS_BAR,
                                                      normalize_embeddings=NORMALIZE_EMBEDDINGS,
                                                      device=device)
    return document_embeddings


def get_split_embeddings(split_articles_df: pd.DataFrame, function_to_apply_on_arrays: Callable):
    split_articles_df_copy = split_articles_df.copy()
    embeddings = get_embeddings(split_articles_df_copy[SPLIT_ARTICLES].tolist())
    split_articles_df_copy[SPLIT_ARTICLES_EMBEDDING] = [embeddings[i, :] for i in range(embeddings.shape[0])]
    split_articles_df_function_applied = split_articles_df_copy.groupby(PARTICIPANT_ID).agg(
        binary_labels=(BINARY_LABELS, 'mean'),
        embeddings=(SPLIT_ARTICLES_EMBEDDING, function_to_apply_on_arrays)).reset_index()
    return split_articles_df_function_applied


def dimensionality_reduction(embedding: np.ndarray):
    pass
