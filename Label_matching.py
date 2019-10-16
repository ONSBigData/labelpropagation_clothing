# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:27:50 2018
@author: martih

Functions to perform string matching between labels and product names and
label products accordingly.
"""
import pandas as pd
from nltk.metrics import distance


def label_prod_desc(labelled_items, item_id):
    """Function to return the unique product descriptions for a given ons item
    Args:
        labelled_item (Dataframe): Pandas data frame containing
                                   the labelled dataset
        item_id (int): Integer number of the ons_item
    Return:
        array: product name of ons item of given item_id
    """
    # find item of given id
    item = labelled_items[labelled_items['item_id'] == item_id]
    return item['Web_product_desc'].unique()


def calculate_ratio(df, prod_desc, metric):
    """Function to calcualte the string comparsiosns ratio combinations
    for all unique item descriptions and product names.

    Args:
        df (Dataframe): Dataframe containg data to be labelled
        prod_desc (list(str): unique product descriptions of labelled data
        metric (function): distance metric function to be used

    Returns:
        (tuple):
            :Dataframe: with calcluated ratio for all unique label names
            :list: column names of ratio values created
    """
    # loop through product descriptions
    for i, lab in enumerate(prod_desc):
        ratio = []
        # calculate ratio each combination
        for name in df.loc[:, 'name']:

            if metric == distance.jaccard_distance:
                ratio.append(metric(set(lab.lower()), set(name.lower())))
            else:
                ratio.append(metric(lab.lower(), name.lower()))
        # append ration to dataframe

        df.loc[:, 'ratio_' + str(i)] = ratio
    # column names containg the ratio values
    cols = df.keys()[(-1*len(prod_desc)):].tolist()
    return df, cols


def find_label_fuzzy(item_id, prod_desc, df, out_col='item_id',
                     threshold=70, greater=True, less=False, neg_thresh=30):
    """
    Function to apply the accepted threshold to the ratio and append labels.
    This function must be run after calcuate_ratio.

    Args:
        item_id (int): integer id value of label to be assigned
        prod_desc (list(str)): unique product descriptions of labelled data
        df (Dataframe): dataframe with data to be labelled, with ratios
                        pre calculated using calcuate_ratio
        threshold (float): threshold value for accepting matches
        greater (bool): True if string matching metric requires greater than
                        value is a better match, false otherwise
                        e.g. fuzzywuzzy matching
        less (bool): True if string matching metric requires lees than a value
                     is a better match, false otherwise
                     e.g. edit distance or jaccard distance

    Return:
        (tuple):
        :Dataframe: Dataframe containing only the matched items
        :Dataframe: origional dataframe with labels appended
    """
    # Empty dataframe to hold mathches
    matched = pd.DataFrame()
    neg_match = pd.DataFrame()

    # For each unique lablled distribution find if any matches have been found
    for i in range(len(prod_desc)):
        # if metric requires greater than threshold as better - fuzzy matching
        if greater:
            matched = matched.append(df.loc[
                          df['ratio_'+str(i)] > threshold], ignore_index=False)

            neg_match = neg_match.append(df.loc[
                         df['ratio_'+str(i)] < neg_thresh], ignore_index=False)

        # if metric requires less than threshold as better - edit or jaccard
        if less:

            matched = matched.append(df.loc[df['ratio_'+str(i)] < threshold],
                                     ignore_index=False)
            neg_match = neg_match.append(df.loc[
                         df['ratio_'+str(i)] > neg_thresh], ignore_index=False)
    # drop any duplicated rows from dataframe (i.e. matched to several labels)

    matched = matched.drop_duplicates()
    neg_match = neg_match.drop_duplicates()

    # find matched items in dataframe and label with itemid

    df.loc[matched.index, out_col] = int(item_id)

    neg_index = set(neg_match.index).difference(set(matched.index))
    df.loc[neg_index, out_col] = 0
    df.loc[df[out_col].isnull(), out_col] = -1

    return matched, df
