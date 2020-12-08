#encoding=utf8


import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

dataset_raw_path = "data/dataset_raw.csv"

def prepare_dataset() :
  out_test = open("data/test.csv", 'w')
  out_train = open("data/train.csv", 'w')
  for line in open(dataset_raw_path, 'r').readlines():
    if random.randint(0, 1):
      out_test.write(line)
    else:
      out_train.write(line)

def create_set(path) :
  setX = []
  setY = []
  for line in open(path, "r").readlines() :
    elem = line.split(",")
    setX.append(re.sub("[^a-zA-Z]", " ", elem[2].replace("#", " ")))
    setY.append(elem[1])
  return setX, setY

def encode_set_TFIDF(set_to_encode) :
  vectorizer = TfidfVectorizer()
  return vectorizer.fit_transform(set_to_encode)
