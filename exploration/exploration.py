from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer


def find_number_of_tokens_in_a_sentence(sentences: List, model_name: str) -> List:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(sentences, add_special_tokens=False)
    number_of_tokens = [len(sentence_tokens_list) for sentence_tokens_list in encoded_input.input_ids]
    return number_of_tokens


def plot_number_of_tokens_histogram(texts_list: List, model_name: str):
    number_of_tokens = find_number_of_tokens_in_a_sentence(texts_list, model_name)
    sns.histplot(number_of_tokens, bins=25)
    plt.title('number of tokens per interview')
    plt.xlabel('number of tokens')
    plt.ylabel('number of occurrences')
    histogram_mean = round(np.array(number_of_tokens).mean(), 1)
    histogram_median = round(float(np.median(np.array(number_of_tokens))), 1)
    histogram_maximum = round(np.max(np.array(number_of_tokens)), 1)
    plt.axvline(histogram_mean, color='red', linestyle='dashed', label=f'mean = {histogram_mean}')
    plt.axvline(histogram_median, color='black', linestyle='dashed', label=f'median = {histogram_median}')
    plt.axvline(histogram_maximum, color='green', linestyle='dashed', label=f'maximum = {histogram_maximum}')
    plt.legend()
    plt.grid()
    plt.show()
