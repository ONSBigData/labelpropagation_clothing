"""

Pipeline to perform both fuzzy matching and then label propagation for
labelling clothing data

The function labelling pipeline is to run the pipeline for a single item category.
"""
from src import clean_most_common, Label_matching, Word_vectors
import logging
import difflib

from nltk.metrics import distance
import fuzzywuzzy.fuzz as fuzz
import sklearn.semi_supervised as semi


# errors
class UnknownMethod(Exception):
    pass


class UnknownParameter(Exception):
    pass


class LabellingError(Exception):
    pass


# set default list of dictionaries, each containing a fuzzy matching function with the parameters to run it
matching_list_default = [
    {
        "label": "label_fuz",
        "func": fuzz.partial_ratio,
        "thresh": 70,
        "neg": 25,
        "greater": True,
        "less": False,
    },
    {
        "label": "label_edit",
        "func": distance.edit_distance,
        "thresh": 30,
        "neg": 75,
        "greater": False,
        "less": True,
    },
    {
        "label": "label_jaccard",
        "func": distance.jaccard_distance,
        "thresh": 0.30,
        "neg": 0.75,
        "greater": False,
        "less": True,
    },
]

word_vectorizer_default = {
    "epochs": 200,
    "print_sim": False,
    "ngram_range": (1, 2),
    "doc2vec_thresh": 0.5,
    "tfidf_thresh": 0.5,
    "count_thresh": 0.5,
    "fast_thresh": 0.5,
    "word2vec_thresh": 0.5,
    "doc2vec_params": {"alpha": 0.01, "kernel": "knn", "n_neighbors": 6},
    "tfidf_params": {"alpha": 0.01, "kernel": "knn", "n_neighbors": 6},
    "count_params": {"alpha": 0.01, "kernel": "knn", "n_neighbors": 6},
    "fast_params": {"alpha": 0.01, "kernel": "knn", "n_neighbors": 6},
    "word2vec_params": {"alpha": 0.01, "kernel": "knn", "n_neighbors": 6},
}


# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def labelling_pipeline(
    data,
    matching_params=None,
    word_vectorizer_params=None,
    n_words=15,
    name_col="name",
    stop_words=None,
    item_id=1,
    label_names=None,
    common_words=True,
    cat_clean=True,
    division=None,
    category=None,
    subcategory=None,
    wv_names=("doc2vec", "tfidf", "count"),
    use_difflib=True,
    notebook=True,
):

    """
    Args:
        data (DataFrame): contains item data 
        matching_params (list): Contains a list of dictionaries with parameters for each fuzzy matching method 
        word_vectorizer_params (dict): Contains parameters for word embeddings
        n_words (int): number of words to use when cleaning most common words
        name_col (str): the column name of the name column on which to do matching and create word vectors
        stop_words (list): words to be removed from product names before matching
        item_id (int): integer for new labels, set to 1 be default
        label_names (list): strings that are the labels for matching
        common_words (bool): flag to clean common words or not
        cat_clean (bool): post label propagation step to clean division
        division (string): name of division to clean
        category (string): category to clean
        subcategory (list(str)): the subcategory words to clean with
        wv_names (list): word embedding methods to use, should be at least one of
                         - "doc2vec, tfidf, count, fast, word2vec"
        use_difflib (bool): forces FuzzyWuzzy to use difflib.SequenceMatcher over python-levenshtein. The latter
            is faster, however will require changing the string matching parameters from those provided as default.
        notebook (bool): switch the tqdm module for the loading bars depending if running in a jupyter notebook or not

    Returns:
        DataFrame: the Cleaned and labelled dataframe
    """

    # import the correct version on tqdm depending on if running from notebook
    if notebook:
        import tqdm.notebook as tqdm
    else:
        import tqdm

    # set to difflib library
    if use_difflib:
        logger.info("Using difflib for sequence matching with fuzzywuzzy")
        fuzz.SequenceMatcher = difflib.SequenceMatcher

    # If true function will clean file keeping those entries which contain
    # most common words
    if common_words:
        names = data[name_col]

        logger.info("Cleaning with common words")
        cleaned = clean_with_common_words(
            data, names, n_words, stop_words, name_col=name_col
        )
    else:
        cleaned = data

    logger.info(f"size of data after clean with common words: {len(cleaned)}")
    # Fuzzy Matching
    # Fuzzy match label names using three distance metrics
    logger.info("Fuzzy Matching")

    # Find matches above acceptance threshold, passing the parameters defined
    for match_dict in tqdm.tqdm(matching_params, desc="Fuzzy matching in progress:"):
        match, cleaned = fuzzy_matching(
            cleaned, label_names, item_id=item_id, name_col=name_col, **match_dict
        )

    # Assign labels where 2 of 3 string metrics agree
    cleaned = assign_labels(
        cleaned, labels=[match_dict["label"] for match_dict in matching_params]
    )

    # perform broad category cleaning for accurate fuzzy labels
    if cat_clean:
        cleaned = clean_labels(cleaned, item_id, division, category, subcategory)

    logger.info(
        f"Number of fuzzyMatch labels: {len(cleaned.loc[cleaned['label_fuzzyMatch'] == 1])}"
    )
    logger.info(
        f"Number of negative labels: {len(cleaned.loc[cleaned['label_fuzzyMatch'] == 0])}"
    )
    logger.info("Word Vectors")

    # create word vectors for all of the types in wv_names
    embeddings_dict = create_word_vectors(
        cleaned, wv_names, name_col=name_col, notebook=True, **word_vectorizer_params
    )

    # loop through the word vectors and perform label propagation
    logger.info("Label propagation")
    for key in tqdm.tqdm(embeddings_dict, desc="Running label propagation"):
        cleaned = run_label_propagation(embeddings_dict, key, cleaned)

    logger.info("Finding modal labels")
    cleaned = get_modal_labels(cleaned, methods=list(embeddings_dict.keys()))

    return cleaned


def clean_with_common_words(data, names, n_words, stop_words, name_col="name"):
    """
    Function to find the most common words in a list of words and then use these common words
    to clean a DataFrame. Only retaining rows which contain the words in the selected name_col.

    Args:
        data (DataFrame): The data that is being cleaned
        names: the words to use in determining the most common words
        n_words: the number of top words to use in cleaning
        stop_words: stopwords to remove when finding the most common words
        name_col: the name of the column in data to clean with the common words

    Returns:
        DataFrame: The cleaned DataFrame
    """

    # find n_words most common words from input names
    common, word_dist = clean_most_common.most_common_words(
        names, n_words, stop=stop_words
    )
    # keep only entries containing the most common words
    data_clean, data_bad = clean_most_common.split_on_common_words(
        common, data[name_col].str.lower()
    )
    # create cleaned dataframe

    return data.loc[data_clean.index]


def fuzzy_matching(cleaned, label_names, item_id=1, name_col="name", **kwargs):
    """
    Function to calculate the fuzzy matching metric for a DataFrame and a given list of labels to match then
    calculate if any of the metrics are above a given matching threshold. Return the data frame with a
    matched flag given by item_id.

    Args:
        cleaned (DataFrame): The data frame to match the labels too
        item_id (int default=1): The integer that will signify that a match has been accepted
        label_names (list(str): a list of labels names to match the data to
        name_col (str): The name of the column that the labels are being matched to
        **kwargs (dict): A dictionary with the fuzzy matching parameters and thresholds

    Returns:
        (DataFrame, DataFrame): a DataFrame containing only the poitive matched products,
                                the whole data with extra match column
    """
    # claculate the metric of each of the label_names
    cleaned, cols1 = Label_matching.calculate_ratio(
        cleaned, label_names, kwargs["func"], name_col=name_col
    )
    # Find matches above acceptance threshold
    match, cleaned = Label_matching.find_label_fuzzy(
        item_id,
        label_names,
        cleaned,
        threshold=kwargs["thresh"],
        neg_thresh=kwargs["neg"],
        out_col=kwargs["label"],
        greater=kwargs["greater"],
        less=kwargs["less"],
    )

    return match, cleaned


def assign_labels(cleaned, labels):
    """
    Once fuzzy matching has been performed calculate the modal label by combining all three metrics
    Args:
        cleaned (DataFrame): The DataFrame
        labels (list(str)): A list containing the names of the columns to be considered in calculating the mode

    Returns:
        DataFrame: The data with an additional column with the mode
    """

    # calculate the mode of the DataFrame columns
    mode = cleaned.loc[:, labels].mode(axis=1)

    # add new column to DataFrame
    cleaned.loc[:, "label_fuzzyMatch"] = mode[0]

    # if the mode has more than one value, switch mode to -1 as uncertain
    if mode.shape[1] > 1:
        cleaned.loc[mode[1].notnull(), "label_fuzzyMatch"] = -1

    return cleaned


def clean_labels(cleaned, item_id, division, category, subcategory):
    """
    Apply some conservative cleaning to the fuzzy matching modal label column.

    Args:
        subcategory list(str): List of subcategory strings which if not present indicate a match is false:
        cleaned (DataFrame): The data to be processed
        item_id (int): the integer indicating a match has been made
        division (str): The string that if not present in the division column indicates the matching is false
        category (str): The string that if not present in the category column indicates the matching is false
        subcategory (list(str)): List of subcategory strings which if not present indicate a match is false:

    Returns:
        DataFrame: The data with matching column cleaned
    """
    # Find items that don't contain division word
    cleaned.loc[
        ((cleaned["label_fuzzyMatch"] == item_id) & (cleaned["division"] != division)),
        "label_fuzzyMatch",
    ] = -1

    # Find items that don't contain category word
    cleaned.loc[
        ((cleaned["label_fuzzyMatch"] == item_id) & (cleaned["category"] != category)),
        "label_fuzzyMatch",
    ] = -1

    # loop to find items that don't contain sub-category word
    if subcategory:
        for j, sub in enumerate(subcategory):
            if j == 0:
                ind = ~(cleaned.subcategory.str.contains(sub))
            else:
                ind *= ~(cleaned.subcategory.str.contains(sub))

        ind *= cleaned.loc[:, "label_fuzzyMatch"] == item_id
        cleaned.loc[ind, ["label_fuzzyMatch"]] = -1

    return cleaned


def create_word_vectors(cleaned, wv_names, name_col="name", notebook=True, **kwargs):
    """
    Create word vector arrays for each of the word vector types in wv_names
    word embedding methods to use, should be at least one of
            - "doc2vec, tfidf, count, fast, word2vec"
    Args:
        cleaned (DataFrame): The data
        wv_names (list(str)): list of strings containing the flags for the word vectors to be created
        name_col (str): the string name of the column from which the word vectors are to be built
        notebook (bool): switch the tqdm module for the loading bars depending if running in a jupyter notebook or not
        **kwargs (dict): The keyword parameters for the word vector methods and label propagation

    Returns:
        dict: Dictionary containing the calculated word vectors for the name column for each
                model type and the parameters for label propagation
    """
    if notebook:
        import tqdm.notebook as tqdm
    else:
        import tqdm
    # Create array of product names for word vectorization
    names = Word_vectors.remove_stopwords(cleaned.loc[:, name_col])

    # output dictionary
    word_vectors_dict = dict()

    # Create word vectors with Doc2vec,TF-IDF and Count vectorizer
    # loop through each vector type and calculate the vectors
    for vectype in tqdm.tqdm(wv_names, desc="Computing vectors:"):
        try:
            if vectype == "doc2vec":
                # create model
                doc2vec_mod = Word_vectors.fit_doc2vec(
                    names, epochs=kwargs["epochs"], print_sim=kwargs["print_sim"]
                )
                # infer vectors
                doc2vec_vec = doc2vec_mod.docvecs.vectors_docs
                # create output dictionary entry
                word_vectors_dict[vectype] = {
                    "mod": doc2vec_mod,
                    "vec": doc2vec_vec,
                    "params": kwargs["doc2vec_params"],
                }
                logger.info(f"calculated {vectype}")

            elif vectype == "tfidf":
                # create model
                tfidf_matrix, tf_mod = Word_vectors.fit_tf_idf(
                    names.values,
                    ngram_range=kwargs["ngram_range"],
                    print_sim=kwargs["print_sim"],
                )
                # create output dictionary
                word_vectors_dict[vectype] = {
                    "vec": tfidf_matrix.toarray(),
                    "mod": tf_mod,
                    "params": kwargs["tfidf_params"],
                }
                logger.info(f"calculated {vectype}")

            elif vectype == "count":
                # create vectors
                count_matrix, count_mod = Word_vectors.count_vectorizer(
                    names.values,
                    ngram_range=kwargs["ngram_range"],
                    print_sim=kwargs["print_sim"],
                )
                # create output dictionary
                word_vectors_dict[vectype] = {
                    "vec": count_matrix.toarray(),
                    "mod": count_mod,
                    "params": kwargs["count_params"],
                }

                logger.info(f"calculated {vectype}")

            elif vectype == "fast":
                # calculate model and vectors
                fast_vec, fast_mod = Word_vectors.fasttext_vectors(names)
                # create output dictionary
                word_vectors_dict[vectype] = {
                    "vec": fast_vec,
                    "mod": fast_mod,
                    "params": kwargs["fast_params"],
                }
                logger.info(f"calculated {vectype}")

            elif vectype == "word2vec":
                # cacluate model and vectors
                w2v_vec, w2v_mod = Word_vectors.fit_word2vec(names)
                # create output dictionary
                word_vectors_dict[vectype] = {
                    "vec": w2v_vec,
                    "mod": w2v_mod,
                    "params": kwargs["word2vec_params"],
                }

                logger.info(f"calculated {vectype}")

            # Raise errors if the vector is not one of the specified types
            else:
                raise UnknownMethod(
                    f"{vectype} not recognised should be one of:"
                    "doc2vec, tfidf, count, fast, word2vec"
                )
        except UnknownMethod:
            logger.warning(UnknownMethod)

    return word_vectors_dict


def run_label_propagation(embeddings_dict, key, cleaned):
    """
    run the label propagation for a word vector matrix

    Args:
        embeddings_dict (dict): dictionary containing required keywords and the word vectors
        key (str): name of the word vector type for naming the column and selecting the right embedding dict entry
        cleaned (DataFrame): the data to append new columns too

    Returns:
        DataFrame: the DataFrame with new column with result of label propagation
    """
    # select parameter for the propagation

    input_vector = embeddings_dict[key]["vec"]
    params = embeddings_dict[key]["params"].copy()

    thresh = params["thresh"]
    # remove threshold from dictionary so that can use in label spreading
    params.pop("thresh")

    # construct the model
    label_mod = semi.label_propagation.LabelSpreading().set_params(**params)
    # fit the model
    label_mod.fit(input_vector, cleaned.loc[:, "label_fuzzyMatch"])

    # label all point -1, then change according to label distributions
    try:
        cleaned.loc[:, f"label_labprop_{key}"] = -1
        cleaned.loc[
            label_mod.label_distributions_[:, 1] >= thresh, f"label_labprop_{key}"
        ] = 1
        cleaned.loc[
            label_mod.label_distributions_[:, 0] >= thresh, f"label_labprop_{key}"
        ] = 0

    except IndexError:
        # if we get an IndexError, it is likely there are not any negative labels
        raise LabellingError(
            "No labels found for at least one class. Check class label counts and try"
            + " adjusting fuzzy matching hyperparameters"
        )

    return cleaned


def get_modal_labels(cleaned, methods):
    """
    calculate the modal label for the label propagation word vectors. If only one vector used don't calculate the mode

    Args:
        cleaned (DataFrame): the data
        methods (list(str): list of word vector methods to find the column name to consider in the mode

    Returns:
        DataFrame: The DataFrame with an extra modal column
    """

    # if more than one word vector is used calculate the mode
    if len(methods) > 1:
        logger.info("More than one method used, finding modal label")
        # find the column names
        labelprop_columns = [f"label_labprop_{method}" for method in methods]

        # calculate the mode
        modal_labels = cleaned[labelprop_columns].mode(axis=1)
        # assign mode to DataFrame
        cleaned["label_labprop_mode"] = modal_labels[0]

        # if more than one mode value it is -1 for uncertain
        if modal_labels.shape[1] > 1:
            cleaned.loc[modal_labels[1].notnull(), "label_labprop_mode"] = -1
    else:
        logger.info(
            f"Only single method used, setting labels for {methods[0]} as modal labels"
        )
        cleaned["label_labprop_mode"] = cleaned[f"label_labprop_{methods[0]}"]

    return cleaned
