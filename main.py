# encoding=utf8


import random
import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


import string
import nltk
import contractions

nltk.download('stopwords')
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

dataset_raw_path = "data/dataset_raw.csv"
train_path = "data/train.csv"
test_path = "data/test.csv"

def encode_set(set_to_encode):
    cv = CountVectorizer(analyzer='word') 
    return cv.fit_transform(set_to_encode)

def prepare_dataset():
    out_test = open(test_path, 'w')
    out_train = open(train_path, 'w')
    for line in open(dataset_raw_path, 'r').readlines() :
        if random.randint(0, 1):
            out_test.write(line)
        else:
            out_train.write(line)


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



def k_nearest_neghbours(trainX, trainY, K, testX, testY) :
  model = KNeighborsClassifier(n_neighbors = K)
  clf = model.fit(trainX, trainY)
  return clf.score(testX, testY)

# TODO : find best K
def knn_score(trainX, trainY, testX, testY) :
  return k_nearest_neghbours(trainX, trainY, 5, testX, testY)


if __name__ == "__main__":
  prepare_dataset()
  trainX, trainY = create_set(train_path)
  testX, testY = create_set(test_path)
  trainX = encode_set(trainX)
  x_train, x_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3)
  # Testing KNN
  print(knn_score(x_train, y_train, x_test, y_test))


