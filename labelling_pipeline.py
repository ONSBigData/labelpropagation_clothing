# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:34:04 2018

@author: martih

Pipeline to perform both fuzzy matching and then label propagation for
labelling clothing data

"""


import importlib
import clean_most_common
import Word_vectors
import Label_matching
import logging
import tqdm

import fuzzywuzzy.fuzz as fuzz
import nltk.metrics.distance as distance
import sklearn.semi_supervised as semi

importlib.reload(Label_matching)

# errors


class UnknownMethod(Exception):
    pass

def labelling_pipeline(data, nwords=15, names=None, stopwords=None,
                       fuzzy_thres=None, edit_thres=None,
                       jacard_thres=None, fuzzy_neg=None, edit_neg=None,
                       jacard_neg=None, item_id=None, label_names=None,
                       param_doc2vec=None, param_tfidf=None, common_words=True,
                       param_count=None, negative_labels=True, n_negative=15,
                       cat_clean=True, division=None, category=None,
                       subcategory=None, doc2vec_thresh=0.5, tfidf_thresh=0.5,
                       count_thresh=0.5, fast_thresh=0.5, word2vec_thresh=0.5,
                       wv_names=['doc2vec', 'tfidf', 'count'], verbose=True):
    """Function pipeline to clean, match to labels and label propogate through
    a dataframe. Function is based on notebook Clean_label_coats_sample and
    functions there in.

    Args:
        data (DataFrame): input dataframe to be cleaned
        nwords (int): number of words to use in cleaning with most common words
        stopwords (list(str)): list of strings to be removed before finding
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

    # set up logging
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if verbose:
        logging.basicConfig(level=logging.INFO, format=log_fmt)
    else:
        logging.basicConfig(level=logging.WARNING, format=log_fmt)
        
    # If true section will clean file keeping those entries which contain
    # most common words

    logging.info('Cleaning with common words')
    cleaned = clean_with_common_words(common_words, data, names, nwords, stopwords)

    logging.info(f'cleaned: {len(cleaned)}')
    # Fuzzy Matching
    # Fuzzy match label names using fuzzy_wuzzy partial ratio
    # calculate closeness metric
    logging.info('Fuzzy Matching')

    # set list of dictionaries, each containing a fuzzy matching function with the parameters to run it
    matching_list = [{"label": "label_fuz",
                      "func": fuzz.partial_ratio,
                      "thresh": fuzzy_thres,
                      "neg": fuzzy_neg,
                      "greater": True,
                      "less": False
                      },
                     {"label": "label_edit",
                      "func": distance.edit_distance,
                      "thresh": edit_thres,
                      "neg": edit_neg,
                      "greater": False,
                      "less": True
                      },
                     {"label": "label_jaccard",
                      "func": distance.jaccard_distance,
                      "thresh": jacard_thres,
                      "neg": jacard_neg,
                      "greater": False,
                      "less": True
                      }
                     ]

    # Find matches above acceptance threshold, passing the parameters define as above
    for match_dict in tqdm.tqdm_notebook(matching_list, desc="Fuzzy matching in progress:"):
        match, cleaned = fuzzy_matching(cleaned, item_id, label_names, **match_dict)

    # Assign labels where 2 of 3 string metrics agree
    cleaned = assign_labels(cleaned,
                            labels=[match_dict["label"] for match_dict in matching_list]
                            )

    # perform broad category cleaning for accurate fuzzy labels
    cleaned = clean_labels(cleaned, cat_clean, item_id, division, category, subcategory)

    logging.info('Word Vectors')

    word_vectorizer_params = {"epochs": 200,
                              "print_sim": False,
                              "ngram_range": (1, 2)
                              }

    embeddings_dict = word_vectors(cleaned, wv_names, **word_vectorizer_params)



    print('Number of fuzzyMatch labels: {}'
          .format(len(cleaned.loc[cleaned['label_fuzzyMatch'] == 1])))
    print('Number of negative labels: {}'
          .format(len(cleaned.loc[cleaned['label_fuzzyMatch'] == 0])))

    # Label propogation
    # Create label propagation labels
    # Labeld accepted if label distribution probabilities are
    # above give threshold, default is 0.5

    print('Label Propogation')


    for vectype in wv_names:
        if vectype == 'doc2vec':
            # doc2vec: first create input vector then perform label
            print('label propogation: dov2vec')
            input_doc2vec = doc2vec_mod.docvecs.vectors_docs
            label_doc2vec = (semi.label_propagation.LabelSpreading()
                                              .set_params(**param_doc2vec))
            label_doc2vec.fit(input_doc2vec,
                              cleaned.loc[:, 'label_fuzzyMatch'])

            # label all point -1, then change according to label distributions
            cleaned.loc[:, 'label_labprop_d2v'] = -1
            cleaned.loc[label_doc2vec.label_distributions_[:, 1] >= doc2vec_thresh,
                        'label_labprop_d2v'] = 1
            cleaned.loc[label_doc2vec.label_distributions_[:, 0] >= doc2vec_thresh,
                        'label_labprop_d2v'] = 0

        if vectype == 'tfidf':
            # Label Propagation:- TFIDF
            print('label propogation: TF-IDF')

            input_tfidf = tfidf_matrix.toarray()
            label_tfidf = semi.label_propagation.LabelSpreading()\
                                           .set_params(**param_tfidf)
            label_tfidf.fit(input_tfidf, cleaned.loc[:, 'label_fuzzyMatch'])

            # Create labels TF-IDF
            cleaned.loc[:, 'label_labprop_tfidf'] = -1
            cleaned.loc[label_tfidf.label_distributions_[:, 1] >= tfidf_thresh,
                        'label_labprop_tfidf'] = 1
            cleaned.loc[label_tfidf.label_distributions_[:, 0] >= tfidf_thresh,
                        'label_labprop_tfidf'] = 0

        if vectype == 'count':
            # Label Propagation Count vectorization
            print('label propogation: Count vectorizer')

            input_count = count_matrix.toarray()
            label_count = semi.label_propagation.LabelSpreading()\
                                           .set_params(**param_count)
            label_count.fit(input_count, cleaned.loc[:, 'label_fuzzyMatch'])

            # create label Count vectorizer
            cleaned.loc[:, 'label_labprop_count'] = -1
            cleaned.loc[label_count.label_distributions_[:, 1] >= count_thresh,
                        'label_labprop_count'] = 1
            cleaned.loc[label_count.label_distributions_[:, 0] >= count_thresh,
                        'label_labprop_count'] = 0

        if vectype == 'word2vec':
            print('label propogation: word2vec')

            label_w2v = semi.label_propagation.LabelSpreading()\
                                         .set_params(**param_doc2vec)
            label_w2v.fit(w2v_vec, cleaned.loc[:, 'label_fuzzyMatch'])

            # create label word2vec vectorizer
            cleaned.loc[:, 'label_labprop_w2v'] = -1
            cleaned.loc[label_w2v.label_distributions_[:, 1] >= word2vec_thresh,
                        'label_labprop_w2v'] = 1
            cleaned.loc[label_w2v.label_distributions_[:, 0] >= word2vec_thresh,
                        'label_labprop_w2v'] = 0

        if vectype == 'fast':
            print('label propogation: fast')

            label_fast = semi.label_propagation.LabelSpreading()\
                                          .set_params(**param_doc2vec)
            label_fast.fit(fast_vec, cleaned.loc[:, 'label_fuzzyMatch'])

            # create label word2vec vectorizer
            cleaned.loc[:, 'label_labprop_fast'] = -1
            cleaned.loc[label_fast.label_distributions_[:, 1] >= fast_thresh,
                        'label_labprop_fast'] = 1
            cleaned.loc[label_fast.label_distributions_[:, 0] >= fast_thresh,
                        'label_labprop_fast'] = 0
    if len(wv_names) >= 2:
        if ('fast' in wv_names) & ('word2vec' in wv_names) & ('doc2vec' in wv_names):
            mode = cleaned[['label_labprop_fast', 'label_labprop_d2v',
                            'label_labprop_w2v']].mode(axis=1)

        elif ('fast' in wv_names) & ('word2vec' in wv_names):
            mode = cleaned[['label_labprop_fast', 'label_labprop_tfidf',
                            'label_labprop_w2v']].mode(axis=1)
        elif 'word2vec' in wv_names:
            mode = cleaned[['label_labprop_w2v', 'label_labprop_tfidf',
                            'label_labprop_count']].mode(axis=1)

        elif 'fast' in wv_names:
            mode = cleaned[['label_labprop_fast', 'label_labprop_tfidf',
                            'label_labprop_count']].mode(axis=1)

        elif 'doc2vec' in wv_names:

            mode = cleaned[['label_labprop_d2v', 'label_labprop_tfidf',
                            'label_labprop_count']].mode(axis=1)
        cleaned.loc[:, 'label_labprop'] = mode[0]

        # if not assigned label as -1 for unlabelled
        if mode.shape[1] > 1:
            cleaned.loc[mode[1].notnull(), 'label_labprop'] = -1
    else:
        if vectype == 'doc2vec':
            cleaned['label_labprop'] = cleaned['label_labprop_d2v']
        elif vectype == 'tfidf':
            cleaned['label_labprop'] = cleaned['label_labprop_tfidf']
        elif vectype == 'count':
            cleaned['label_labprop'] = cleaned['label_labprop_count']
        elif vectype == 'fast':
            cleaned['label_labprop'] = cleaned['label_labprop_fast']
        elif vectype == 'word2vec':
            cleaned['label_labprop'] = cleaned['label_labprop_w2v']
        else:
            print('vector type unknown')

    return cleaned


def clean_with_common_words(common_words, data, names, nwords, stopwords):

    if common_words:
        # find n_words most common words from input names
        common, word_dist = clean_most_common.most_common_words(names,
                                                                nwords,
                                                                stop=stopwords)
        # keep only entries containing the most common words
        data_clean, data_bad = clean_most_common.split_on_commonwords(common, data['name'].str.lower())
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
                                                     less=kwargs["neg"])

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


def word_vectors(cleaned, wv_names, **kwargs):

    # Create array of product names for word vectorization
    names = Word_vectors.remove_stopwords(cleaned.loc[:, 'name'])

    # Create word vectors with Doc2vec,TF-IDF and Count vectorizer
    word_vectors_dict = dict()

    for vectype in tqdm.tqdm_notebook(wv_names, desc="Computing vectors:"):
        try:
            if vectype == 'doc2vec':
                doc2vec_mod = Word_vectors.fit_Doc2vec(names, epochs=kwargs["epochs"],
                                                       print_sim=kwargs["print_sim"])
                word_vectors_dict[vectype] = {"mod": doc2vec_mod}
                logging.info(f"calculated {vectype}")

            elif vectype == 'tfidf':
                tfidf_matrix, tf_mod = Word_vectors.TF_IDF(names.values,
                                                           ngram_range=kwargs["ngram_range"],
                                                           print_sim=kwargs["print_sim"])
                word_vectors_dict[vectype] = {"matrix": tfidf_matrix,
                                              "mod": tf_mod}
                logging.info(f"calculated {vectype}")

            elif vectype == 'count':
                count_matrix, count_mod = Word_vectors\
                                            .count_vectorizer(names.values,
                                                              ngram_range=kwargs["ngram_range"],
                                                              print_sim=kwargs["print_sim"])
                word_vectors_dict[vectype] = {"matrix": count_matrix,
                                              "mod": count_mod}

                logging.info(f"calculated {vectype}")

            elif vectype == 'fast':
                fast_vec, fast_mod = Word_vectors.fasttext_vectors(names)
                word_vectors_dict[vectype] = {"vec": fast_vec,
                                              "mod": fast_mod}
                logging.info(f"calculated {vectype}")

            elif vectype == 'word2vec':

                w2v_vec, w2v_mod = Word_vectors.fit_word2vec(names)
                word_vectors_dict[vectype] = {"vec": w2v_vec,
                                              "mod": w2v_mod}

                logging.info(f"calculated {vectype}")

            else:
                raise UnknownMethod(f"{vectype} not recognised should be one of:"
                                    + "doc2vec, tfidf, count, fast, word2vec")
        except UnknownMethod:
            logging.warning(UnknownMethod)

    return word_vectors_dict


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