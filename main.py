# encoding=utf8


import random
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#for testing
import pandas as pd
import matplotlib.pyplot as plt

import string
import nltk
import contractions

nltk.download('stopwords')
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()
tfdidf_vectorizer = TfidfVectorizer(analyzer="word")

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
    for line in open(path, "r").readlines():
        elem = line.split(",", maxsplit=2)
        set_x.append(normalisation(elem[2]))
        set_y.append(True if elem[1] == "1" else False)

    return set_x, set_y



def knn_predict(trainX, trainY, test_string) :
  model = KNeighborsClassifier(n_neighbors = 1)
  clf = model.fit(trainX, trainY)
  test= tfdidf_vectorizer.transform([normalisation(test_string)])
  return model.predict(test)[0]


def knn_find_best_k(trainX, trainY, testX, testY) :
  scores = []
  for i in range(1, 35) :
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(trainX, trainY)
    scores.append(model.score(testX, testY))
  plt.plot(scores)
  plt.show()

# best K = 1
def knn_score(trainX, trainY, testX, testY) :
  model = KNeighborsClassifier(n_neighbors = 1)
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
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
  test_tweet = "beautiful sign by vendor"
  # Testing KNN
  print("KNN score = ", knn_score(x_train, y_train, x_test, y_test))
  print("Predicted", knn_predict(x_train, y_train, test_tweet), " for \"", test_tweet, "\"")
  # Testing SVM
  print("SVM score = ", svm_score(x_train, y_train, x_test, y_test))
  print("Predicted", svm_predict(x_train, y_train, test_tweet), " for \"", test_tweet, "\"")


