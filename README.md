# Semi-supervised machine learning with word embeddings for classification in price statistics

### Code to accompany the paper Semi-supervised machine learning with word embedding for classification in price statistics: Martindale et.al 

Here we make available the code to perform the same analysis as is described in the paper along with a 
mock dataset. 

The mock data is made by selecting the 100 most common words from the actual product names and then choosing 
between 5 and 15 words to form new product names. 
The resulting product names do not make sense as names but aim is for the mock data 
to be some what representative of the true data. 
In addition to the product name we also randomly select a broad and narrow retailer category for
 the product from those available in the real data. 

## Running the code
All code in this project is written using Python 3.7 and the required python models can be installed using
the [requirements file](requirements.txt). 

In order to run the pipeline use the [ipython notebook](label_propagation_and_classification.ipynb).
A [static rendering](https://onsbigdata.github.io/labelpropagation_clothing/) of the notebook is also provided.

## The source files to run this notebooks are:
* [clean_most_common.py](src/clean_most_common.py) - functions to fine the most common words in the text and select only products containing these words
* [Label_matching.py](src/Label_matching.py) - functions to perform fuzzy matching with fuzzywuzzy, edit distance and Jaccard ratio
* [labelling_pipeline.py](src/labelling_pipeline.py) - Function to perform the matching and label propagation for 1 category item
* [Word_vectors.py](src/Word_vectors.py) - Function to calculate the word vectors for the product names

## The datafiles:
* [clothing_mock_data.csv](data/clothing_mock_data.csv) - The mock data set to be labelled
* [seed_labels.json](data/seed_labels.json) - A json file containing a dictionary of starting labels for the fuzzy matching and label propagation for each category

