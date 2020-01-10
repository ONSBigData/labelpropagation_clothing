"""
Functions to create word vectors from product names using three different
methods.
    * Doc2vec - See gensim package for further detail
    * fasttext - See gensim package for further detail
    * word2vec - See gensim package for further detail
    * TF-IDF vectorizer - using Scikit-learn implementation
    * Count vectorizer - using scikit-learn implementation
"""


import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from gensim.models.fasttext import FastText
from gensim.models import Word2Vec


def remove_stopwords(product_name, rem_punctuation=True, other_list=None):
    """
    Function to remove stopwords from product names. By default will remove
    stopwords from scikit-learn standard list but extra stopwords can be appended.

    Args:
        product_name (Series): series containing the product names
                               to be vectorized
        rem_punctuation (bool): it true punctuation is removed from product names,
                         Default is True
        other_list (list(str)): if exists extra stop words are appended to the
                               list from scikit-learn.

    Return:
        Series: Series with product names after removing stopwords
    """

    # sklearn stopwords
    stop = stop_words.ENGLISH_STOP_WORDS
    # lowercase
    product_name = product_name.str.lower()
    # remove digits
    product_name = product_name.str.replace("\d", "")
    # optionally remove punctuation
    if rem_punctuation:
        product_name = product_name.str.replace("[^\w\s]", " ")
    # optionally add additional stopwords
    if other_list:
        stop = stop.union(set(other_list))
    # remove stopwords from product names
    product_name = product_name.apply(
        lambda x: " ".join([item for item in x.split() if item not in stop])
    )

    return product_name


class LabeledLineSentence(object):
    """
    Class to create tagged document structure for doc2vec.
    Product names are added to a tagged document structure with a tag
    equivalent to the product name for ease of finding.
    Product names are tokenized as required for doc2vec.
    """

    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        # Create the tagged document object required for doc2vec
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[str(self.labels_list[idx])])


def fit_doc2vec(
    product_name,
    print_sim=True,
    vector_size=12,
    window=12,
    min_count=1,
    workers=4,
    alpha=0.025,
    min_alpha=0.001,
    epochs=100,
):
    """
    Function to run Doc2vec on product names and return model. See gensim
    doc2vec documentation for further details on the doc2vec parameters.

    Args:
        product_name (Series): product names to be vectorised
        print_sim (bool): If true vector similarities printed to the screen
        vector_size (int): size of word vector to pass to Doc2vec
        window (int): size of window to pass to Doc2vec
        min_count (int): count of words below which to ignore to
                         pass to Doc2vec
        workers (int): number of cores to use to pass to Doc2vec
        alpha (float): learning rate to pass to Doc2vec
        min_alpha (float): minimum learning rate to pass to Doc2vec
        epochs (int): number of passes over text corpora to pass to Doc2vec

    Return:
        object: Doc2vec model
    """

    # create tagged document list
    it = LabeledLineSentence(product_name.values, product_name.index)
    # define model
    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        alpha=alpha,
        min_alpha=min_alpha,
    )

    # introduce vocab to the model
    model.build_vocab(it)

    # train doc2Vec model epochs are number of iterations to perform
    model.train(it, total_examples=len(product_name), epochs=epochs)

    # if desired print sample vector similarity - useful to check quality
    # of model during testing
    if print_sim:
        print(it.doc_list[2], ", ", it.doc_list[12])
        print("Vector similarity: ", model.docvecs.similarity("2", "12"), "\n")

        print(it.doc_list[150], ", ", it.doc_list[154])
        print("Vector similarity: ", model.docvecs.similarity("150", "154"), "\n")

        print(it.doc_list[4], ", ", it.doc_list[417])
        print("Vector similarity: ", model.docvecs.similarity("4", "417"), "\n")

    return model


def fit_tf_idf(
    product_name,
    print_sim=True,
    stop_words="english",
    ngram_range=(1, 2),
    analyzer="word",
    min_df=1,
    max_features=None,
):
    """
    Function to produce term-frequency inverse-document-frequency vectors for
    product names.

    Args:
        product_name (Series): product names to be vectorised
        print_sim (Bool): If true vector cosine seminaries will be printed
        stop_words (str): Stopwords to use pass to TfidfVectorizer
                          see TfidfVectorizer documentation
        ngram_range (tuple): The lower and upper boundary of the range
                             of n-values for different n-grams to be extracted.
                             See TfidfVectorizer documentation.
       analyzer (string): Whether the feature should be made of word or
                          character n-grams. See TfidfVectorizer documentation.
       min_df (int): When building the vocabulary ignore terms that have a
                     document frequency strictly lower than the given
                     threshold. See TfidfVectorizer documentation.
       max_features (int or None): If not None, build a vocabulary that only consider the top
                                   max_features ordered by term frequency across the corpus.
                                   See scikit-learn documentation

    Return:
        (array, object):
            :tfidf vector array
            :tfidf fitted model
    """
    # create tfidf object
    tf = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        analyzer=analyzer,
        min_df=min_df,
        max_features=max_features,
    )
    # build vectors
    tfidfmatrix = tf.fit_transform(product_name)

    # Optionally print vector similarity to check model quality during testing
    if print_sim:
        print(
            product_name[2],
            " ",
            product_name[12],
            ": ",
            cosine_similarity(tfidfmatrix[2], tfidfmatrix[12])[0],
        )
        print(
            product_name[150],
            " ",
            product_name[154],
            ": ",
            cosine_similarity(tfidfmatrix[150], tfidfmatrix[154])[0],
        )
        print(
            product_name[413],
            " ",
            product_name[417],
            ": ",
            cosine_similarity(tfidfmatrix[413], tfidfmatrix[417])[0],
        )
        print(
            product_name[77],
            " ",
            product_name[85],
            ": ",
            cosine_similarity(tfidfmatrix[77], tfidfmatrix[85])[0],
        )

    return tfidfmatrix, tf


def count_vectorizer(
    product_name,
    print_sim=True,
    analyzer="word",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=None,
):
    """
    Function to produce count frequency vectors for product names.

    Args:
        product_name (Series): product names to be vectorised
        print_sim (Bool): If true vector cosine similarities will be printed
        stop_words (str): Stopwords to use pass to CountVectorizer
                          see CountVectorizer documentation
        ngram_range (tuple): The lower and upper boundary of the range
                             of n-values for different n-grams to be extracted.
                             See CountVectorizer documentation.
       analyzer (string): Whether the feature should be made of word or
                          character n-grams. See CountVectorizer documentation.
       max_features (int or None): If not None, build a vocabulary that only consider the top
                                   max_features ordered by term frequency across the corpus.
                                   See scikit-learn documentation
    Return:
        (tuple):
        :array: count vectorizer array
        :object: fitted count vectorizer model
    """
    # create count vectorizer object
    count_vectorizer = CountVectorizer(
        analyzer=analyzer,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    # build vectors
    count_matrix = count_vectorizer.fit_transform(product_name)

    # Optionally print vector similarity to check model quality during testing
    if print_sim:
        print(
            product_name[2],
            " ",
            product_name[12],
            ": ",
            cosine_similarity(count_matrix[2], count_matrix[12])[0],
        )
        print(
            product_name[150],
            " ",
            product_name[154],
            ": ",
            cosine_similarity(count_matrix[150], count_matrix[154])[0],
        )
        print(
            product_name[413],
            " ",
            product_name[417],
            ": ",
            cosine_similarity(count_matrix[413], count_matrix[417])[0],
        )
        print(
            product_name[77],
            " ",
            product_name[85],
            ": ",
            cosine_similarity(count_matrix[77], count_matrix[85])[0],
        )

    return count_matrix, count_vectorizer


def fasttext_vectors(product_name, size=100, window=5, min_count=3, n_iter=5):
    """
    Function to calculate fastText vectors for product names.
    see gensim documentation
    Args:
        product_name (Series): product names to be vectorised
        size (int): length of the vectors
        window(int): window function size
        min_count (int): the minimum count frequency for a word to be included
        n_iter (int): number of iterations/epochs passes over data

    Return:
        (array, object): array of vectors and the trained model
    """
    product_name = product_name.str.split(" ")

    # train the model
    model_fasttext = FastText(
        product_name,
        size=size,
        window=window,
        min_count=min_count,
        sg=1,
        hs=1,
        iter=n_iter,
        negative=10,
    )

    # infer vectors
    i = 0
    vectors = []
    for nlist in product_name:
        i += 1
        doc = []
        for word in nlist:
            if word in model_fasttext:
                doc.append(model_fasttext[word])
        if len(doc) == 0:
            print(doc)

        doc = list(np.array(doc).mean(axis=0))
        vectors.append(doc)
    vectors = np.array(vectors)
    return vectors, model_fasttext


def fasttext_infvec(product_name, model_fasttext):
    """
    Infer the fastText vectors for new words
    Args:
        product_name (Series): product names to be vectorised
        model_fasttext (object): a gensim fasttext model object
    return:
        array: the inferred word vectors for the products
    """
    product_name = product_name.str.split(" ")

    # loop through product names and infer vector
    i = 0
    vectors = []
    for n_list in product_name:
        i += 1
        doc = []
        for word in n_list:
            if word in model_fasttext:
                doc.append(model_fasttext[word])
        if len(doc) == 0:
            print(doc)

        doc = list(np.array(doc).mean(axis=0))
        vectors.append(doc)
    vectors = np.array(vectors)
    return vectors


def fit_word2vec(
    product_name,
    vector_size=12,
    window=12,
    min_count=1,
    workers=4,
    min_alpha=0.001,
    epochs=100,
):
    """
    Function to calculate word2vec vectors for product names.
    see gensim documentation
    Args:
        product_name (Series): product names to be vectorised
        vector_size (int): length of the vectors
        window (int): window function size
        min_count (int): the minimum count frequency for a word to be included
        workers (int): number of worker nodes to use when training
        min_alpha (float): the minimum learning rate to decrease to
        epochs (int): number of iterations/epochs passes over data

    Return:
        (array, object): array of vectors and trained model
    """

    product_name = product_name.str.split(" ")
    # train model
    model_word2vec = Word2Vec(
        product_name,
        size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        min_alpha=min_alpha,
        iter=epochs,
    )

    # infer vector for each product name
    i = 0
    vectors = []
    for nlist in product_name:

        i += 1
        doc = []
        for word in nlist:
            if word in model_word2vec.wv.vocab.keys():

                doc.append(model_word2vec.wv.get_vector(word))
        if len(doc) == 0:
            print(doc)

        doc = list(np.array(doc).mean(axis=0))
        vectors.append(doc)
    vectors = np.array(vectors)

    return vectors, model_word2vec


def word2vec(product_name, model_word2vec):
    """
    Infer the fastText vectors for new words
    Args:
        product_name (Series): product names to be vectorised
        model_word2vec (object): a gensim word2vec model object
    return:
        array: the inferred word vectors for the products
    """
    product_name = product_name.str.split(" ")

    i = 0
    vectors = []
    # loop through the product names inferring the vectors in tern
    for n_list in product_name:

        i += 1
        doc = []
        for word in n_list:
            if word in model_word2vec.wv.vocab:
                doc.append(model_word2vec.wv.get_vector(word))
        if len(doc) == 0:
            print(i)
            print(doc)

            doc = np.nan
        else:
            doc = list(np.array(doc).mean(axis=0))
        vectors.append(doc)
    vectors = np.array(vectors)
    return vectors
