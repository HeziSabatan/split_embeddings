from typing import List, Union

import pandas as pd
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from configurations.data_frame_columns import TEXTS, SPLIT_ARTICLES

enc = tiktoken.get_encoding("r50k_base")


def length_function(text: str) -> int:
    return len(enc.encode(text))


def split_articles_according_to_token_limit(text: str, token_chunk_size: int, token_chunk_overlap: int,
                                            max_number_of_splits: Union[int, None] = None,
                                            min_num_words: Union[int, None] = None) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=token_chunk_size,
        chunk_overlap=token_chunk_overlap,
        length_function=length_function,
    )
    splits = splitter.split_text(text)
    if max_number_of_splits is not None:
        splits = [splits[i] for i in range(min(len(splits), max_number_of_splits))]
    if min_num_words is not None:
        splits = [split_text for split_text in splits if len(split_text.split()) > min_num_words]
    return splits


def get_split_article_df(data_df: pd.DataFrame, token_chunk_size: int, token_chunk_overlap: int,
                         max_number_of_splits: Union[int, None] = None,
                         min_num_words: Union[int, None] = None) -> pd.DataFrame:
    data_df_copy = data_df.copy()
    data_df_copy[SPLIT_ARTICLES] = data_df[TEXTS].apply(
        lambda text: split_articles_according_to_token_limit(text, token_chunk_size, token_chunk_overlap,
                                                             max_number_of_splits, min_num_words))
    data_df_all_splits = data_df_copy.explode(SPLIT_ARTICLES)
    return data_df_all_splits
