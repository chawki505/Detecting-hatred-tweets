# encoding=utf8


import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

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


def encode_set_TFIDF(set_to_encode):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(set_to_encode)


def prepare_dataset():
    out_test = open("data/test.csv", 'w')
    out_train = open("data/train.csv", 'w')
    for line in open(dataset_raw_path, 'r').readlines():
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
        set_y.append(elem[1])

    return set_x, set_y



def k_nearest_neghbours(trainX, trainY, K) :
  model = KNeighborsClassifier(n_neighbors = k)
  model.fit(trainX, trainY)
