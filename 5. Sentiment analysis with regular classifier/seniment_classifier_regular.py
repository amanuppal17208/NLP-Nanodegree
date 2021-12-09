import os
import glob
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
import nltk
import pickle
import joblib
nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords # import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB


# stemmer = SnowballStemmer("english")
# lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()
clf1 =GaussianNB()

cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

#Function to load the data seperately
def preprocess_data_file(file_path):

    try:
        with open(os.path.join(file_path), "rb") as f:
            cache_data = pickle.load(f)
        print("Read preprocessed data from cache file:", file_path)
    except:
        pass

    words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                          cache_data['words_test'], cache_data['labels_train'],
                                                          cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                              cache_data['words_test'], cache_data['labels_train'],
                                                              cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test

def prepare_imdb_data(data,labels):
    """Prepare training and test sets from IMDb movie reviews."""

    # TODO: Combine positive and negative reviews and labels

    data_train = data['train']['pos'] + data['train']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']


    data_train, labels_train, data_test, labels_test = shuffle(data_train, labels_train, data_test, labels_test,
                                                               random_state=10)

    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test

def review_to_words(review):
    """Convert a raw review string into a sequence of words."""

    # Remove HTML tags and non-letters,
    #       convert to lowercase, tokenize,
    #       remove stopwords and stem
    soup = BeautifulSoup(review, "html5lib")
    text = soup.get_text()
    text = text.lower()
    token = simple_preprocess(text)
    token = [word for word in token if word not in stopwords.words("english") and len(word) > 3]
    # token = [stemmer.stem(lemmatizer.lemmatize(word, pos='v')) for word in token]
    # Return final list of words
    token = [stemmer.stem(word) for word in token]
    words = token

    return words

def read_imdb_data(data_dir='../data/imdb-reviews'):
    """Read IMDb movie reviews from given directory.

    Directory structure expected:
    - data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/

    """

    # Data, labels to be returned in nested dicts matching the dir. structure
    data = {}
    labels = {}

    # Assume 2 sub-directories: train, test
    for data_type in ['train', 'test']:

        data[data_type] = {}
        labels[data_type] = {}

        # Assume 2 sub-directories for sentiment (label): pos, neg
        for sentiment in ['pos', 'neg']:

            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            # Fetch list of files for this sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            # Read reviews data and assign labels
            for f in files:
                with open(f,encoding="utf-8") as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)

    return data, labels

def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=cache_dir, cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # TODO: Fit a vectorizer to training documents and use it to transform them
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(decode_error='ignore', preprocessor=lambda x: x, min_df=15, max_df=0.2,
                                     max_features=vocabulary_size, tokenizer=lambda x: x)
        features_train = vectorizer.fit_transform(np.array(words_train))
        features_train = features_train.toarray()

        # TODO: Apply the same vectorizer to transform the test documents (ignore unknown words)
        features_test = vectorizer.transform(words_test)
        features_test = features_test.toarray()

        # NOTE: Remember to convert the features using .toarray() for a compact representation

        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                              vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                                                     cache_data['features_test'], cache_data['vocabulary'])

    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary

if __name__=='__main__':

    data,labels=read_imdb_data()
    data_train, data_test, labels_train, labels_test = prepare_imdb_data(data, labels)
    words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, labels_test)
    features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test)
    features_train = normalize(features_train,axis=1,norm='max')
    features_test = normalize(features_test,axis=1,norm='max')
    clf1.fit(features_train, labels_train)

    print("[{}] Accuracy: train = {}, test = {}".format(
        clf1.__class__.__name__,
        clf1.score(features_train, labels_train),
        clf1.score(features_test, labels_test)))