# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:34:04 2018

@author: martih

Pipeline to perform both fuzzy matching and then label propagation for
labelling clothing data

The function labelling pipeline is to run the pipeline for a single item category.
This function

"""

import importlib
import clean_most_common
import Word_vectors
import Label_matching
import logging
import tqdm.notebook as tqdm
import difflib

import fuzzywuzzy.fuzz as fuzz
import sklearn.semi_supervised as semi

from nltk.metrics import distance

importlib.reload(Label_matching)

# variables
# TODO: move to config file

# set list of dictionaries, each containing a fuzzy matching function with the parameters to run it
matching_list = [{"label": "label_fuz",
                  "func": fuzz.partial_ratio,
                  "thresh": 70,
                  "neg": 25,
                  "greater": True,
                  "less": False
                  },
                 {"label": "label_edit",
                  "func": distance.edit_distance,
                  "thresh": 30,
                  "neg": 75,
                  "greater": False,
                  "less": True
                  },
                 {"label": "label_jaccard",
                  "func": distance.jaccard_distance,
                  "thresh": 0.30,
                  "neg": 0.75,
                  "greater": False,
                  "less": True
                  }
                 ]

word_vectorizer_params = {"epochs": 200,
                          "print_sim": False,
                          "ngram_range": (1, 2),
                          "doc2vec_thresh": 0.5,
                          "tfidf_thresh": 0.5,
                          "count_thresh": 0.5,
                          "fast_thresh": 0.5,
                          "word2vec_thresh": 0.5,
                          "doc2vec_params": {'alpha': 0.01, 'kernel': 'knn', 'n_neighbors': 6},
                          "tfidf_params": {'alpha': 0.01, 'kernel': 'knn', 'n_neighbors': 6},
                          "count_params": {'alpha': 0.01, 'kernel': 'knn', 'n_neighbors': 6},
                          "fast_params": {'alpha': 0.01, 'kernel': 'knn', 'n_neighbors': 6},
                          "word2vec_params": {'alpha': 0.01, 'kernel': 'knn', 'n_neighbors': 6}

                          }

# errors


class UnknownMethod(Exception):
    pass


class UnknownParameter(Exception):
    pass


class LabellingError(Exception):
    pass


def labelling_pipeline(data, matching_params=matching_list,
                       word_vectorizer_params=word_vectorizer_params,
                       n_words=15, names=None, stop_words=None,
                        item_id=None, label_names=None, common_words=True,
                       cat_clean=True, division=None, category=None,
                       subcategory=None, wv_names=("doc2vec", "tfidf", "count"),
                       verbose=True,
                       use_difflib=True ):

        """Function pipeline to clean, match to labels and label propogate through
        a dataframe. Function is based on notebook Clean_label_coats_sample and
        functions there in.

        Args:
            data (DataFrame): input dataframe to be cleaned
            n_words (int): number of words to use in cleaning with most common words
            stop_words (list(str)): list of strings to be removed before finding
                                   most common words
            fuzzy_thres (float): threshold for fuzzy matching to be accepted
            edit_thres (float): threshold for edit distance matching to be accepted
            jacard_thres (float): threshold for jacard matching to be accepted
            item_id (int): integer for the new labels, usually 1
            label_names (list(str)): list of strings that are the labels
                                     for matching
            param_doc2vec (dict): dictionary with the parameters for label
                                  propogation with doc2vec
            param_tfidf (dict): dictionary with parameters for label propogation
                                with tfidf vectors
            param_count (dict): dictionary with parameters for label propogation
                                with count vectors
            common_words (bool): If True will clean dataframe with common words
            negative_labels (bool): if True will randomly create negative labels
                                    for label propogation
            cat_clean (bool): If true will perform step after label propogation
                              to clean a given division name from labels
            division (str): String name of division to clean
            category (str): string name of categories to clean
            doc2vec_thresh (float): threshold propoabilty for label to be 1 in
                                    label propogation doc2vec vectors
            tfidf_thresh (float): threshold propoabilty for label to be 1 in
                                  label propogation tfidf vectors
            count_thresh (float): threshold propoabilty for label to be 1 in
                                  label propogation count vectors

        Returns:
            (tuple):
            :DataFrame: Cleaned and labelled dataframe
            :array: Array containing doc2vec vectors
            :array: Array containing TF-IDF vectors
            :array: Array containing count vectors
        """

        # set to difflib libarary
        if use_difflib:
            logging.info("Using difflib for sequence matching with fuzzywuzzy")
            fuzz.SequenceMatcher = difflib.SequenceMatcher

        # set up logging
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if verbose:
            logging.basicConfig(level=logging.INFO, format=log_fmt)
        else:
            logging.basicConfig(level=logging.WARNING, format=log_fmt)

        # If true section will clean file keeping those entries which contain
        # most common words

        logging.info('Cleaning with common words')
        cleaned = clean_with_common_words(common_words, data, names, n_words, stop_words)

        logging.info(f'cleaned: {len(cleaned)}')
        # Fuzzy Matching
        # Fuzzy match label names using fuzzy_wuzzy partial ratio
        # calculate closeness metric
        logging.info('Fuzzy Matching')

        # TODO: Consider refactoring to loop within rather than over the function, similar to run_label_propagation()
        # Find matches above acceptance threshold, passing the parameters defined
        for match_dict in tqdm.tqdm(matching_params, desc="Fuzzy matching in progress:"):
            match, cleaned = fuzzy_matching(cleaned, item_id, label_names, **match_dict)

        # Assign labels where 2 of 3 string metrics agree
        cleaned = assign_labels(cleaned,
                                labels=[match_dict["label"] for match_dict in matching_params]
                                )

        # perform broad category cleaning for accurate fuzzy labels
        cleaned = clean_labels(cleaned, cat_clean, item_id, division, category, subcategory)

        logging.info(f"Number of fuzzyMatch labels: {len(cleaned.loc[cleaned['label_fuzzyMatch'] == 1])}")
        logging.info(f"Number of negative labels: {len(cleaned.loc[cleaned['label_fuzzyMatch'] == 0])}")
        logging.info('Word Vectors')

        if len(cleaned.loc[cleaned['label_fuzzyMatch'] == 0]) == 0:
            try:
                embeddings_dict = create_word_vectors(cleaned, wv_names, **word_vectorizer_params)

            except IndexError:
                # if we get an IndexError, it is likely there are not any negative labels
                raise LabellingError("No labels found for at least one class. Check class label counts and try"
                                     + " adjusting fuzzy matching hyperparameters")

        else:
            embeddings_dict = create_word_vectors(cleaned, wv_names, **word_vectorizer_params)

        logging.info("Label propagation")
        cleaned = run_label_propagation(embeddings_dict, cleaned)

        logging.info("Finding modal labels")
        cleaned = get_modal_labels(cleaned, methods=list(embeddings_dict.keys()))

        return cleaned


def clean_with_common_words(common_words, data, names, n_words, stop_words):

    if common_words:
        # find n_words most common words from input names
        common, word_dist = clean_most_common.most_common_words(names,
                                                                n_words,
                                                                stop=stop_words)
        # keep only entries containing the most common words
        data_clean, data_bad = clean_most_common.split_on_common_words(common, data['name'].str.lower())
        # create cleaned dataframe

        return data.loc[data_clean.index]
    else:
        return data


def fuzzy_matching(cleaned, item_id, label_names, **kwargs):

    cleaned, cols1 = Label_matching.calculate_ratio(cleaned, label_names,
                                                    kwargs["func"])
    # Find matches above acceptance threshold
    match, cleaned = Label_matching.find_label_fuzzy(item_id, label_names,
                                                     cleaned,
                                                     threshold=kwargs["thresh"],
                                                     neg_thresh=kwargs["neg"],
                                                     out_col=kwargs["label"],
                                                     greater=kwargs["greater"],
                                                     less=kwargs["less"])

    return match, cleaned


def assign_labels(cleaned, labels):

    mode = cleaned.loc[:, labels].mode(axis=1)

    cleaned.loc[:, 'label_fuzzyMatch'] = mode[0]

    if mode.shape[1] > 1:
        cleaned.loc[mode[1].notnull(), 'label_fuzzyMatch'] = -1

    return cleaned


def clean_labels(cleaned, cat_clean, item_id, division, category, subcategory):

    if cat_clean:
        cleaned.loc[((cleaned['label_fuzzyMatch'] == item_id) &
                     (cleaned['division'] != division)),
                    'label_fuzzyMatch'] = -1

        cleaned.loc[((cleaned['label_fuzzyMatch'] == item_id) &
                    (cleaned['category'] != category)),
                    'label_fuzzyMatch'] = -1
        if subcategory:
            for j, sub in enumerate(subcategory):
                if j == 0:
                    ind = ~(cleaned.subcategory.str.contains(sub))
                else:
                    ind *= ~(cleaned.subcategory.str.contains(sub))

            ind *= (cleaned.loc[:, 'label_fuzzyMatch'] == item_id)
            cleaned.loc[ind, ['label_fuzzyMatch']] = -1

    return cleaned


def create_word_vectors(cleaned, wv_names, **kwargs):

    # Create array of product names for word vectorization
    names = Word_vectors.remove_stopwords(cleaned.loc[:, 'name'])

    # Create word vectors with Doc2vec,TF-IDF and Count vectorizer
    word_vectors_dict = dict()

    for vectype in tqdm.tqdm(wv_names, desc="Computing vectors:"):
        try:
            # TODO: tidy up if-else; I think there should be a better way of doing this
            if vectype == 'doc2vec':
                doc2vec_mod = Word_vectors.fit_Doc2vec(names, epochs=kwargs["epochs"],
                                                       print_sim=kwargs["print_sim"])
                doc2vec_vec = doc2vec_mod.docvecs.vectors_docs
                word_vectors_dict[vectype] = {"mod": doc2vec_mod,
                                              "vec": doc2vec_vec,
                                              "thresh": kwargs["doc2vec_thresh"],
                                              "params": kwargs["doc2vec_params"]}
                logging.info(f"calculated {vectype}")

            elif vectype == 'tfidf':
                tfidf_matrix, tf_mod = Word_vectors.TF_IDF(names.values,
                                                           ngram_range=kwargs["ngram_range"],
                                                           print_sim=kwargs["print_sim"])
                word_vectors_dict[vectype] = {"vec": tfidf_matrix.toarray(),
                                              "mod": tf_mod,
                                              "thresh": kwargs["tfidf_thresh"],
                                              "params": kwargs["tfidf_params"]}
                logging.info(f"calculated {vectype}")

            elif vectype == 'count':
                count_matrix, count_mod = Word_vectors\
                                            .count_vectorizer(names.values,
                                                              ngram_range=kwargs["ngram_range"],
                                                              print_sim=kwargs["print_sim"])
                word_vectors_dict[vectype] = {"vec": count_matrix.toarray(),
                                              "mod": count_mod,
                                              "thresh": kwargs["count_thresh"],
                                              "params": kwargs["count_params"]}

                logging.info(f"calculated {vectype}")

            elif vectype == 'fast':
                fast_vec, fast_mod = Word_vectors.fasttext_vectors(names)
                word_vectors_dict[vectype] = {"vec": fast_vec,
                                              "mod": fast_mod,
                                              "thresh": kwargs["fast_thresh"],
                                              "params": kwargs["fast_params"]}
                logging.info(f"calculated {vectype}")

            elif vectype == 'word2vec':

                w2v_vec, w2v_mod = Word_vectors.fit_word2vec(names)
                word_vectors_dict[vectype] = {"vec": w2v_vec,
                                              "mod": w2v_mod,
                                              "thresh": kwargs["word2vec_thresh"],
                                              "params": kwargs["word2vec_params"]}

                logging.info(f"calculated {vectype}")

            else:
                raise UnknownMethod(f"{vectype} not recognised should be one of:"
                                    "doc2vec, tfidf, count, fast, word2vec")
        except UnknownMethod:
            logging.warning(UnknownMethod)

    return word_vectors_dict


def run_label_propagation(embeddings_dict, cleaned):

    # TODO: consider refactoring to loop over rather than within the function as with fuzzy_matching()
    for key in tqdm.tqdm(embeddings_dict, desc="Running label propagation"):

        input_vector = embeddings_dict[key]["vec"]
        thresh = embeddings_dict[key]["thresh"]
        params = embeddings_dict[key]["params"]

        label_mod = semi.label_propagation.LabelSpreading().set_params(**params)

        label_mod.fit(input_vector, cleaned.loc[:, 'label_fuzzyMatch'])

        # label all point -1, then change according to label distributions
        cleaned.loc[:, f'label_labprop_{key}'] = -1
        cleaned.loc[label_mod.label_distributions_[:, 1] >= thresh,
                    f'label_labprop_{key}'] = 1
        cleaned.loc[label_mod.label_distributions_[:, 0] >= thresh,
                    f'label_labprop_{key}'] = 0

    return cleaned


def get_modal_labels(cleaned, methods):

    if len(methods) > 1:
        logging.info("More than one method used, finding modal label")
        labelprop_columns = [f"label_labprop_{method}" for method in methods]
        modal_labels = cleaned[labelprop_columns].mode(axis=1)

        cleaned["label_labprop_mode"] = modal_labels[0]

        # if not assigned label as -1 for unlabelled
        if modal_labels.shape[1] > 1:
            cleaned.loc[modal_labels[1].notnull(), "label_labprop_mode"] = -1
    else:
        logging.info(f"Only single method used, setting labels for {methods[0]} as modal labels")
        cleaned['label_labprop_mode'] = cleaned[f'label_labprop_{methods[0]}']

    return cleaned


def create_negative_labels(cleaned, negative_labels, division, n_negative, category):

    # If required create sample of 20 'negative' labels from the 0 label class
    if negative_labels and not division:
        cleaned.loc[cleaned[cleaned['true_label'] == 0].sample(n_negative).index,
                    ['label', 'label_fuzzyMatch']] = 0
        # cleaned[cleaned['label_fuzzyMatch'] == 0]
    elif negative_labels and division and category:
        cleaned.loc[(cleaned[(cleaned['division'] != division)
                     ].sample(int(n_negative / 2.)).index),
                    ['true_label', 'label_fuzzyMatch']] = 0
        cleaned.loc[(cleaned[(cleaned['category'] != category)
                     ].sample(int(n_negative)).index),
                    ['true_label', 'label_fuzzyMatch']] = 0

    # else:
    #     cleaned.loc[cleaned['true_label'] == 0, 'label_fuzzyMatch'] = 0

    return cleaned