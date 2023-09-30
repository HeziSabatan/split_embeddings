import numpy as np
import pandas as pd

from configurations.model_configurations import TOKEN_CHUNK_SIZE, TOKEN_CHUNK_OVERLAP, MAX_NUMBER_OF_SPLITS, \
    MIN_NUM_WORDS
from configurations.paths import DEPRESSION_DATA_SET_FILE
from models.models import get_split_embeddings
from preprocessing.preprocessing import get_split_article_df


def average_arrays(arrays):
    # Use np.mean to calculate the average
    return np.mean(arrays, axis=0)


if __name__ == '__main__':
    data = pd.read_csv(DEPRESSION_DATA_SET_FILE, delimiter=',')
    data = data.iloc[:3, 1:]
    # plot_number_of_tokens_histogram(data[TEXTS].tolist(), SENTENCE_TRANSFORMER_MODEL)
    split_articles_df = get_split_article_df(data,
                                             token_chunk_size=TOKEN_CHUNK_SIZE,
                                             token_chunk_overlap=TOKEN_CHUNK_OVERLAP,
                                             max_number_of_splits=MAX_NUMBER_OF_SPLITS,
                                             min_num_words=MIN_NUM_WORDS)
    # plot_number_of_tokens_histogram(split_articles_df[SPLIT_ARTICLES].tolist(), SENTENCE_TRANSFORMER_MODEL)

    split_articles_mean = get_split_embeddings(split_articles_df,
                                               function_to_apply_on_arrays=average_arrays)
    split_articles_mean.to_excel('something.xlsx', index=None)
