import copy
import os
import re
import string

import pandas as pd

def add_headers(full_pth='nlpwdl2021_data/thedeep.small.test.txt'):
    '''
    Function to append header to the text files

    Arguments:
    :full_pth: full path to the text file, str

    Return:
    Saves processed file in the root folder
    '''
    sep, ext = os.path.splitext(full_pth)
    sep = sep.split('/')

    with open(full_pth, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.split('\n')
    text.insert(0, 'id, text, label') # append headers, final file is readable by pandas

    with open(f'{sep[1]}_processed.txt', 'w+', encoding="utf-8") as f:
        for sentence in text:
            f.write(sentence + '\n')

# Cleaning and tokenizing text

def lemmatize(text, nlp):
    '''
    Function to lemmatize the text string and convert it into list of lemmas (tokens).

    '''
    doc = nlp(text)
    lemma_list = [token.lemma_ for token in doc]  # lemmatize the text (tokenization and normalization)
    return lemma_list

def process(text, nlp):
    '''
    Function for text preprocessing. Removes stop-words, punctuation, numbers and converts everything
    except abbreviations to lower case.

    '''
    lemma_list = lemmatize(text, nlp)

    # remove stopwords
    filtered_sentence = []
    for word in lemma_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)

    # remove punctuation
    punctuations = string.punctuation
    for word in filtered_sentence:
        if word in punctuations:
            filtered_sentence.remove(word)

    # convert to lower case
    for word in filtered_sentence:
        if not bool(re.search(r'[A-Z]{2,}', word)):  # avoid converting abbreviations to lower case
            filtered_sentence[filtered_sentence.index(word)] = word.lower()

    # substitute numbers
    for word in filtered_sentence:
        token = re.sub(r'(\d+[,.-]*)', '', word)
        token = re.sub(r'[-,.]+', '', token)
        filtered_sentence[filtered_sentence.index(word)] = token

    # remove remaining punctuation and whitespaces
    filtered_sentence = [t for t in filtered_sentence if len(t) >= 2]
    return filtered_sentence


# Creating a vocabulary

def create_count_vocab(series: pd.Series):
    counts = {}

    if type(series) == pd.Series:
        series = series.values

    for text in series:
        for word in text:
            if word not in counts.keys():
                counts[word] = 1
            else:
                counts[word] += 1

    vocab = [x for x in counts.keys()]
    return vocab, counts


def substitute_oov(series: pd.Series, vocab: list, counts: dict, threshold:int):
    '''
    Function to remove OOV words from vocabulary, as well as from text.
    '''
    series = copy.deepcopy(series)
    vocab = copy.deepcopy(vocab)
    vocab_to_remove = []

    for sentence_idx in range(len(series)):
        for word in series[sentence_idx]:
            if counts[word] < threshold:
                vocab_to_remove.append(word)
                series[sentence_idx] = list(filter(lambda a: a != word, series[sentence_idx]))

    vocab_to_remove = set(vocab_to_remove)
    vocab = [word for word in vocab if word not in vocab_to_remove]
    return series, vocab
