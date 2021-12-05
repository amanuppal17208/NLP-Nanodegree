import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models

nltk.download('wordnet')
np.random.seed(400)

stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):

    result = []
    for token in simple_preprocess(text):

        if token not in STOPWORDS and len(token) > 3:

            result.append(lemmatize_stemming(token))

    return result

if __name__=='__main__':

    data = pd.read_csv('abcnews-date-text.csv')
    data_text = data[:300000][['headline_text']]
    data_text['index'] = data_text.index
    documents = data_text
    processed_docs = documents['headline_text'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.1,keep_n=10000)
    bow_corpus = [dictionary.doc2bow(sentence) for sentence in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10,
                                           id2word=dictionary,
                                           passes=2)
    for idx, topic in lda_model.print_topics(-1):

        print("Topic: {} \nWords: {}".format(topic, idx))
        print("\n")

    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10,
                                                 id2word=dictionary,
                                                 passes=2)

    for idx, topic in lda_model_tfidf.print_topics(-1):

        print("Topic: {} Word: {}".format(idx, topic))
        print("\n")

    unseen_document = "My favorite sports activities are running and swimming."

    # Data preprocessing step for the unseen document
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))

    for index, score in sorted(lda_model[bow_vector], key=lambda tup: tup[1], reverse=True):

        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
