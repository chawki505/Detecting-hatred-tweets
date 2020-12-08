# encoding=utf8


import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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

    # remove http and url
    url_reg = re.compile('http(s)?://\S+|www\.\S+')
    text = url_reg.sub(r'', text)

    # remove punctuations
    punctuations_reg = re.compile('[%s]' % re.escape(string.punctuation))
    text = punctuations_reg.sub(r'', text)

    # remove new line characters
    text = text.replace('\n', '')

    # remove Emojis
    emoji_reg = re.compile("["u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    text = emoji_reg.sub(r'', text)

    # remove stopwords
    text = ' '.join([elem for elem in text.split() if elem not in mystopwords])

    # Lemmatization
    text = ' '.join([lemm.lemmatize(word) for word in text.split()])

    # remove user
    text = text.replace("user", "")

    return text


def create_set(path):
    set_x = []
    set_y = []
    for line in open(path, "r").readlines():
        elem = line.split(",")

        set_x.append(normalisation(elem[2]))
        set_y.append(elem[1])

        return set_x, set_y


if __name__ == '__main__':
    pass
