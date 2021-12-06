#In this example, we will use Bayesian Inference to build a Spam detection module in Python.

#Importng all the necessary Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Initiating instances of models
naive_bayes = MultinomialNB()
count_vector = CountVectorizer(stop_words='english')


def return_value(word):
    if word == 'ham':

        return 0

    else:

        return 1

if __name__=='__main__':

    df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'sms_message'])
    df['label'] = df.label.map(return_value)
    X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                        df['label'],
                                                        random_state=1)

    print('Number of rows in the total set: {}'.format(df.shape[0]))
    print('Number of rows in the training set: {}'.format(X_train.shape[0]))
    print('Number of rows in the test set: {}'.format(X_test.shape[0]))

    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)
    naive_bayes.fit(training_data, y_train)
    predictions = naive_bayes.predict(testing_data)

    print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
    print('Precision score: ', format(precision_score(y_test, predictions)))
    print('Recall score: ', format(recall_score(y_test, predictions)))
    print('F1 score: ', format(f1_score(y_test, predictions)))