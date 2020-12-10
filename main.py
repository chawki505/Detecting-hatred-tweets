# encoding=utf8

import sys
import re

# models
from nltk import pos_tag, word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# for testing
import matplotlib.pyplot as plt
import numpy as np

import string
import nltk
import contractions

nltk.download('stopwords', download_dir="cache_nltk/", quiet=True)
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')

nltk.download('wordnet', download_dir="cache_nltk/", quiet=True)
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', download_dir="cache_nltk/", quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir="cache_nltk/", quiet=True)

lemm = WordNetLemmatizer()
tfdidf_vectorizer = TfidfVectorizer(analyzer="word", max_features=7000)

dataset_raw_path = "data/dataset_raw.csv"

knn_model = KNeighborsClassifier(n_neighbors=1)
nb_model = MultinomialNB(alpha=0.1)
svm_model = SVC(C=10)


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

    # remove stopwords
    text = ' '.join([elem for elem in text.split() if elem not in mystopwords])

    # Lemmatization
    text = ' '.join(
        [lemm.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v']
         else lemm.lemmatize(i) for i, j in pos_tag(word_tokenize(text))])

    # remove user
    text = text.replace("user", "")

    # remove Emojis
    emoji_reg = re.compile("[^A-Za-z ]+")
    text = emoji_reg.sub(r'', text)

    # remove spaces
    text = text.split()
    text = ' '.join([i for i in text])

    return text


def create_set(path):
    set_x = []
    set_y = []
    for line in open(path, "r").readlines()[1:]:
        elem = line.split(",", maxsplit=2)
        set_x.append(normalisation(elem[2]))
        set_y.append(True if elem[1] == "1" else False)
    return set_x, set_y


# fonction init model naive bayes
def nb_init(test_x, test_y):
    nb_model.fit(test_x, test_y)


# fonction init model knn
def knn_init(test_x, test_y):
    knn_model.fit(test_x, test_y)


# fonction init model scm
def svm_init(test_x, test_y):
    svm_model.fit(test_x, test_y)


# call init for all model
def init_all(test_x, test_y):
    nb_init(test_x, test_y)
    knn_init(test_x, test_y)
    svm_init(test_x, test_y)


# calcul score naive bayes with best alpha = 0.1
def nb_score(test_x, test_y):
    return nb_model.score(test_x, test_y)


# calcul score knn with best k = 1
def knn_score(test_x, test_y):
    return knn_model.score(test_x, test_y)


# calculate score svm with best C = 10
def svm_score(test_x, test_y):
    return svm_model.score(test_x, test_y)


# function de prediction pour naive bayes
def nb_predict(test_string):
    test = tfdidf_vectorizer.transform([normalisation(test_string)])
    return nb_model.predict(test)[0]


# function de prediction pour knn
def knn_predict(test_string):
    test = tfdidf_vectorizer.transform([normalisation(test_string)])
    return knn_model.predict(test)[0]


# function de prediction pour svm
def svm_predict(test_string):
    test = tfdidf_vectorizer.transform([normalisation(test_string)])
    return svm_model.predict(test)[0]


def knn_find_best_k(test_x, test_y):
    scores = []
    for i in range(1, 35):
        scores.append(knn_model.score(test_x, test_y))
    plt.plot(scores)
    plt.show()


# function pour trouver la meilleur valeur de alpha pour notre model avec plot
def nb_find_best_alpha(test_x, test_y):
    scores = []
    alpha = []
    for i in np.arange(0.1, 1.1, 0.1):
        scores.append(nb_model.score(test_x, test_y))
        alpha.append(i)
    plt.plot(alpha, scores)
    plt.show()


def svm_find_best_c(train_x, train_y):
    param_grid_for_grid_search = {'kernel': ['rbf'], 'C': [1, 10]}
    model = GridSearchCV(svm_model, param_grid_for_grid_search)
    model.fit(train_x, train_y)
    # print best parameter after tuning
    print(model.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(model.best_estimator_)


if __name__ == "__main__":
    X, Y = create_set(dataset_raw_path)
    X = encode_set(X)

    test_tweet = sys.argv[1] if len(
        sys.argv) > 1 else "@user why not @user mocked obama for being black.  @user @user @user @user #brexit"

    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    # init_all(x_train, y_train)

    knn_init(x_train, y_train)
    # Testing KNN
    print("Testing KNN")
    print("KNN score = ", knn_score(x_test, y_test))
    print("Predicted", knn_predict(test_tweet), " for \"", test_tweet, "\"")

    nb_init(x_train, y_train)
    # Testing MB
    print("Testing NB")
    print("NB score = ", nb_score(x_test, y_test))
    print("Predicted", nb_predict(test_tweet), " for \"", test_tweet, "\"")

    # svm_init(x_train, y_train)
    # Testing SVM
    # print("Testing SVM")
    # print("SVM score = ", svm_score(x_test, y_test))
    # print("Predicted", svm_predict(test_tweet), " for \"", test_tweet, "\"")
