"""
Functions to perform string matching between labels and product names and
label products accordingly.
"""
import pandas as pd
from nltk.metrics import distance


def label_prod_desc(labelled_items, item_id):
    """Function to return the unique product descriptions for a given ons item

    Args:
        labelled_items (DataFrame): Pandas data frame containing
                                   the labelled dataset
        item_id (int): Integer number of the ons_item
    Return:
        array: product name of ons item of given item_id
    """
    # find item of given id
    item = labelled_items[labelled_items["item_id"] == item_id]
    return item["Web_product_desc"].unique()


def calculate_ratio(df, prod_desc, metric, name_col="name"):
    """Function to calculate the string comparisons ratio combinations
    for all unique item descriptions and product names.

    Args:
        df (DataFrame): DataFrame containing data to be labelled
        prod_desc (list(str): unique product descriptions of labelled data
        metric (function): distance metric function to be used
        name_col (str): name of the column to match to

    Returns:
        (DataFrame, list):
            :data with calculated ratio for all unique label names
            :list of column names of ratio values created
    """
    # loop through product descriptions
    for i, lab in enumerate(prod_desc):
        ratio = []
        # calculate ratio each combination
        for name in df.loc[:, name_col]:

            if metric == distance.jaccard_distance:
                ratio.append(metric(set(lab.lower()), set(name.lower())))
            else:
                ratio.append(metric(lab.lower(), name.lower()))

        # append ration to DataFrame
        df.loc[:, "ratio_" + str(i)] = ratio

    # column names containing the ratio values
    cols = df.keys()[(-1 * len(prod_desc)) :].tolist()
    return df, cols


def find_label_fuzzy(
    item_id,
    prod_desc,
    df,
    out_col="item_id",
    threshold=70,
    greater=True,
    less=False,
    neg_thresh=30,
):
    """
    Function to apply the accepted threshold to the ratio and append labels.
    This function must be run after calculate_ratio.

    Args:
        item_id (int): integer id value of label to be assigned
        prod_desc (list(str)): unique product descriptions of labelled data
        df (DataFrame): DataFrame with data to be labelled, with ratios
                        pre calculated using calculate_ratio
        out_col (str): the name of the output column containing the matches
        threshold (float): threshold value for accepting matches
        greater (bool): True if string matching metric requires greater than
                        value is a better match, false otherwise
                        e.g. FuzzyWuzzy matching
        less (bool): True if string matching metric requires lees than a value
                     is a better match, false otherwise
                     e.g. edit distance or Jaccard distance
        neg_thresh (float): the threshold for 'accepting' a match as not the item
                            a negative match
    Return:
        (DataFrame, DataFrame):
        :DataFrame containing only the matched items
        :original DataFrame with labels appended
    """
    # Empty DataFrame to hold matches
    matched = pd.DataFrame()
    neg_match = pd.DataFrame()

    # For each unique labeled distribution find if any matches have been found
    for i in range(len(prod_desc)):
        # if metric requires greater than threshold as better - fuzzy matching
        if greater:
            matched = matched.append(
                df.loc[df["ratio_" + str(i)] > threshold], ignore_index=False
            )

            neg_match = neg_match.append(
                df.loc[df["ratio_" + str(i)] < neg_thresh], ignore_index=False
            )

        # if metric requires less than threshold as better - edit or Jaccard
        if less:

            matched = matched.append(
                df.loc[df["ratio_" + str(i)] < threshold], ignore_index=False
            )
            neg_match = neg_match.append(
                df.loc[df["ratio_" + str(i)] > neg_thresh], ignore_index=False
            )

    # drop any duplicated rows from DataFrame (i.e. matched to several labels)
    matched = matched.drop_duplicates()
    neg_match = neg_match.drop_duplicates()

    # find matched items in DataFrame and label with itemid
    df.loc[matched.index, out_col] = int(item_id)

    # find the negative items
    neg_index = set(neg_match.index).difference(set(matched.index))
    df.loc[neg_index, out_col] = 0
    df.loc[df[out_col].isnull(), out_col] = -1

    return matched, df
