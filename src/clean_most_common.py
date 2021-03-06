"""
Functions to clean a DataFrame of unwanted products using most common word.
"""

import pandas as pd
import numpy as np
from matplotlib import colors

from sklearn.feature_extraction import stop_words

from nltk import tokenize
from nltk import FreqDist


def build_stopwords(colour=True, other_list=None):
    """
    Function to return list of stopwords with the option to include colour.
    Colours are from matplotlib colour name list, other stopwords
    from sci-kit learn.

    Args:
        colour (bool): Boolean to tell if colour is included in stoplist,
                       default is true
        other_list (list): List of additional stopwords to append

    Return:
        set: Set containing full list of stopwords
    """

    stop = stop_words.ENGLISH_STOP_WORDS
    # if colour is used append to stopwords
    if colour:
        color_list = colors.cnames.keys()
        stop = stop.union(set(color_list))
    # if other list exists append to stopwords
    if other_list:
        stop = stop.union(set(other_list))

    return stop


def clean_names(names, rem_punctuation=True, rem_stop=True, stop=None, colour=True):

    """ Function to clean the product names, i.e. remove digits, punctuation,
    stopwords and ensure lowercase.
    Args:
        names (Series): Panda series containing the product names to be cleaned
        rem_punctuation (bool): If true punctuation is removed from the product names.
        rem_stop (bool): If true stopwords are built from build_stopwords and
                         removed from product names.
        stop (list(str)): list of additional stopwords to remove,
                          default ==True
        colour (bool): if true colours will be removed with stopwords,
                       default==True
    Return:
        Series: Series containing cleaned product names
    """
    # lower case
    names = names.str.lower()
    # remove digits
    names = names.str.replace("\d", "")
    # optionally remove punctuation
    if rem_punctuation:
        names = names.str.replace("[^\w\s]", " ")
    # optionally remove stopwords
    if rem_stop:
        stop = build_stopwords(other_list=stop, colour=colour)
        names = names.apply(
            lambda x: " ".join([item for item in x.split() if item not in stop])
        )

    return names


def most_common_words(
    names, n_words=15, rem_punctuation=True, rem_stop=True, stop=None, colour=True
):
    """
    Function to find and return the n most common words. The function contains
    options to remove punctuation and stopwords.

    Args:
        names (list): list of the product names to find the most
                      common words from.
        n_words (int): number of most common words to be found.
                      Default is 15, found to be optimum for
                      cleaning coats sample data.
        rem_punctuation (bool): If true remove punctuation before finding most frequent
        rem_stop (bool): If true remove stopwords before finding most frequent
        stop (list): List of stopwords to remove, if None will use stopwords
                     from build_stopwords with default section.
        colour (bool): If true colour names are appended to stopwords
    Return:
        (list(tuple), object):
            : list of tuples with the most common word and
                          its frequency
            : nltk word_dist object with information on word distribution
    """
    # clean product names
    names = clean_names(
        names,
        rem_punctuation=rem_punctuation,
        rem_stop=rem_stop,
        stop=stop,
        colour=colour,
    )
    # fine unique words
    words = tokenize.word_tokenize(names.str.cat(sep=" "))
    # Create word frequency information
    word_dist = FreqDist(words)
    common = word_dist.most_common(n_words)

    return common, word_dist


def split_on_common_words(common, names):
    """
    Function to split names containing common words from those that don't.
    data_clean has files with names containing most common words, data_bad
    are names not containing common words.

    Args:
        common (list(tuple)): most common words from most_common_words
        names (Series): series containing names to be split

    Return:
        (tuple):
            :data_clean: (series) containing correct items,
            :data_bad: (series) containing unwanted items
    """
    # empty series for output data
    data_clean = pd.Series()
    for word in common:
        # append data containing stop word
        data_clean = data_clean.append(names[names.str.contains(word[0])])
        # remove entries just added to data_clean
        names = names.drop(names[names.str.contains(word[0])].index)

    # resulting data without any of the common words
    data_bad = names
    return data_clean, data_bad


def most_common_roc(data, names, word_dist, n_max=200):
    """
    Function to calculate an roc curve for cleaning with most common words,
    varying the number of words used in the cleaning. This allows us to see
    how well the cleaning does and also determine which number of words to use.

    Warning:
        This function currently only works for womens_coats in the toy dataset!

    Args:
        data (DataFrame): DataFrame to be cleaned
        names (list(str)): product names to be cleaned
        word_dist (object): nltk word distribution object
                            from most_common_words
        n_max (int): maximum number of words to use as most common

    Returns:
        (array, array, array):
         :true positive rate,
         :false positive rate,
         :threshold of N-words for cleaning)
    """
    # empty array to collect variables
    tp, tn, fp, fn = [], [], [], []
    # loop through range of n-words used to clean
    for n in range(1, n_max):

        # split on n most common words
        common = word_dist.most_common(n)
        data_clean, data_bad = split_on_common_words(common, names)
        coats_bad = data.loc[data_bad.index]
        coats_good = data.loc[data_clean.index]

        # cacluate and append true positves etc
        tp.append(len(coats_good[coats_good["Type"] == "Womens_Coats"]))
        fp.append(
            len(coats_good[coats_good["Type"] == "Non_Coats"])
            + len(coats_good[coats_good["Type"] == "Non_Clothes"])
        )

        tn.append(
            len(coats_bad[coats_bad["Type"] == "Non_Coats"])
            + len(coats_bad[coats_bad["Type"] == "Non_Clothes"])
        )

        fn.append(len(coats_bad[coats_bad["Type"] == "Womens_Coats"]))

    # calculate true/false postive rate for roc curve
    tpr = np.array([0])
    fpr = np.array([0])
    tpr = np.append(tpr, np.array(tp) / (np.array(tp) + np.array(fn)))
    fpr = np.append(fpr, np.array(fp) / (np.array(fp) + np.array(tn)))
    n_range = np.arange(0, n_max + 1)

    return tpr, fpr, n_range
