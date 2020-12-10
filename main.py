# encoding=utf8

import sys
import re

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# for testing
import matplotlib.pyplot as plt
import numpy as np

import string
import nltk
import contractions

nltk.download('stopwords')
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()
tfdidf_vectorizer = TfidfVectorizer(analyzer="word", max_features=7000)

dataset_raw_path = "data/dataset_raw.csv"


def encode_set(set_to_encode):
    return tfdidf_vectorizer.fit_transform(set_to_encode)


def normalisation(text: str) -> str:
    # make str low
    text = text.lower()

    # remove html caractere
    html_reg = re.compile("<[^<]+?>")
    text = html_reg.sub(r'', text)

    # remove contraction
    text = contractions.fix(text)

    # make str low
    text = text.lower()

    # remove http and url
    url_reg = re.compile('http(s)?://\S+|www\.\S+')
    text = url_reg.sub(r'', text)

    # remove punctuations
    punctuations_reg = re.compile('[%s]' % re.escape(string.punctuation))
    text = punctuations_reg.sub(r'', text)

    # remove new line characters
    text = text.replace('\n', '')

    # TODO : remove space
    text = text.split()
    text = ' '.join([i for i in text if i != '' or i != ' '])

    # remove stopwords
    text = ' '.join([elem for elem in text.split() if elem not in mystopwords])

    # Lemmatization
    text = ' '.join([lemm.lemmatize(word) for word in text.split()])

    # remove user
    text = text.replace("user", "")

    # remove Emojis
    emoji_reg = re.compile("[^A-Za-z ]+")
    text = emoji_reg.sub(r'', text)

    return text


def create_set(path):
    set_x = []
    set_y = []
    for line in open(path, "r").readlines()[1:]:
        elem = line.split(",", maxsplit=2)
        set_x.append(normalisation(elem[2]))
        set_y.append(True if elem[1] == "1" else False)

    return set_x, set_y


def knn_predict(trainX, trainY, test_string):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(trainX, trainY)
    test = tfdidf_vectorizer.transform([normalisation(test_string)])
    return model.predict(test)[0]


# fonction pour predire le label d'un tweet
def nb_predict(trainX, trainY, test_string):
    model = MultinomialNB(alpha=0.1)
    model.fit(trainX, trainY)
    test = tfdidf_vectorizer.transform([normalisation(test_string)])
    return model.predict(test)[0]


def knn_find_best_k(trainX, trainY, testX, testY):
    scores = []
    for i in range(1, 35):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(trainX, trainY)
        scores.append(model.score(testX, testY))
    plt.plot(scores)
    plt.show()


# fonction pour trouver la meilleur valeur de alpha pour notre model avec plot
def nb_find_best_alpha(trainX, trainY, testX, testY):
    scores = []
    alpha = []
    for i in np.arange(0.1, 1.1, 0.1):
        model = MultinomialNB(alpha=i)
        model.fit(trainX, trainY)
        scores.append(model.score(testX, testY))
        alpha.append(i)
    plt.plot(alpha, scores)
    plt.show()


# best K = 1
def knn_score(trainX, trainY, testX, testY):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(trainX, trainY)
    return model.score(testX, testY)


# best alpha = 0.1
def nb_score(trainX, trainY, testX, testY):
    model = MultinomialNB(alpha=0.1)
    model.fit(trainX, trainY)
    return model.score(testX, testY)


def svm_predict(trainX, trainY, test_string) :
  model = SVC()
  model.fit(trainX, trainY)
  test= tfdidf_vectorizer.transform([normalisation(test_string)])
  return model.predict(test)[0]

def svm_score(trainX, trainY, testX, testY) :
  model = SVC()
  model.fit(trainX, trainY)
  return model.score(testX, testY)


if __name__ == "__main__":
    X, Y = create_set(dataset_raw_path)
    X = encode_set(X)

    test_tweet = sys.argv[1] if len(
        sys.argv) > 1 else "@user why not @user mocked obama for being black.  @user @user @user @user #brexit"

    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    print("Avant clean : ", test_tweet)
    print("Apres clean : ", normalisation(test_tweet))


    # Testing KNN
    print("Testing KNN")
    print("KNN score = ", knn_score(x_train, y_train, x_test, y_test))
    print("Predicted", knn_predict(x_train, y_train, test_tweet), " for \"", test_tweet, "\"")
    # Testing SVM
    print("Testing SVM")
    print("SVM score = ", svm_score(x_train, y_train, x_test, y_test))
    print("Predicted", svm_predict(x_train, y_train, test_tweet), " for \"", test_tweet, "\"")
    # Testing MB
    print("Testing NB")
    print("NB score = ", nb_score(x_train, y_train, x_test, y_test))
    print("Predicted", nb_predict(x_train, y_train, test_tweet), " for \"", test_tweet, "\"")
