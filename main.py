# encoding=utf8

import sys
import re
from os import path

# for testing
import matplotlib.pyplot as plt
import numpy as np

# models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from nltk import pos_tag, word_tokenize

# utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import string
import nltk
import contractions

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

mystopwords = stopwords.words('english')

nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

lemm = WordNetLemmatizer()
tfdidf_vectorizer = TfidfVectorizer()

dataset_raw_path = "data/dataset_raw.csv"
dataset_clean_path = "data/dataset_clean.csv"

CONST_BEST_K = 1
CONST_BEST_C = 10
CONST_BEST_ALPHA = 0.2

knn_model = KNeighborsClassifier(n_neighbors=CONST_BEST_K)
nb_model = MultinomialNB(alpha=CONST_BEST_ALPHA)
svm_model = SVC(C=CONST_BEST_C, kernel='linear')


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

    # remove usergi
    text = text.replace("user", "")

    # remove Emojis
    emoji_reg = re.compile("[^A-Za-z ]+")
    text = emoji_reg.sub(r'', text)

    # remove spaces
    text = text.split()
    text = ' '.join([i for i in text])

    return text


def read_dataset(input_file):
    set_x = []
    set_y = []
    for line in open(input_file, "r").readlines()[1:]:
        elem = line.strip('\n').split(",", maxsplit=2)
        set_x.append(elem[2])
        set_y.append(True if elem[1] == "1" else False)
    return set_x, set_y


def create_clean_dataset(input_path, output_path):
    file_ouput = open(output_path, "w")
    file_ouput.writelines("id,label,tweet\n")

    for line in open(input_path, "r").readlines()[1:]:
        elem = line.split(",", maxsplit=2)
        file_ouput.write('{},{},{}\n'.format(elem[0], elem[1], normalisation(elem[2])))
    file_ouput.close()


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


def algo_crossval_score(clf, x, y, subset_number=5):
    return cross_val_score(clf, x, y, cv=subset_number).mean()


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


def knn_find_best_k(train_x, train_y, test_x, test_y):
    scores = []
    k = []
    for i in range(1, 35):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(train_x, train_y)
        scores.append(round(model.score(test_x, test_y) * 100, 2))
        k.append(i)
    plt.plot(k, scores)
    plt.title("Best \"K\" search graph")
    plt.ylabel("Score %")
    plt.xlabel("K value")
    plt.show()


# function pour trouver la meilleur valeur de alpha pour notre model avec plot
def nb_find_best_alpha(train_x, train_y, test_x, test_y):
    scores = []
    alpha = []
    for i in np.arange(0.1, 1.1, 0.1):
        model = MultinomialNB(alpha=i)
        model.fit(train_x, train_y)
        scores.append(round(model.score(test_x, test_y) * 100, 2))
        alpha.append(i)
    plt.plot(alpha, scores)
    plt.title("Best \"alpha\" search graph")
    plt.ylabel("Score %")
    plt.xlabel("Alpha value")
    plt.show()


def svm_find_best_c(train_x, train_y):
    param_grid_for_grid_search = {'kernel': ['rbf'], 'C': [1, 10]}
    svm = SVC()
    model = GridSearchCV(svm, param_grid_for_grid_search)
    model.fit(train_x, train_y)
    # print best parameter after tuning
    print(model.best_params_)
    # print how our model looks after hyper-parameter tuning
    print(model.best_estimator_)


def show_algorithms_comparison(without_cv, with_cv):
    labels = ['K Nearest Neighbors', 'Naive Bayes', 'Support Vector Machine']

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, without_cv, width, label='Without Cross validation')
    rects2 = ax.bar(x + width / 2, with_cv, width, label='With Cross validation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Show algorithms comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":

    if not path.exists(dataset_clean_path):
        if path.exists(dataset_raw_path):
            print("\nCreating clean dataset ...")
            create_clean_dataset(dataset_raw_path, dataset_clean_path)
            print("\t - creating clean dataset : done !")
        else:
            print("\nError dataset raw not found !")
            exit(1)

    print("\nCreating set X, Y ...")
    X, Y = read_dataset(dataset_clean_path)
    print("\t - creating set X, Y: done !")

    print("\nEncoding set X ...")
    X = encode_set(X)
    print("\t - encoding set X: done !")

    test_tweet = sys.argv[1] if len(
        sys.argv) < 1 else "@user why not @user mocked obama for being black.  @user @user @user @user #brexit"

    print("\nSpliting X in train and test ...")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    print("\t - train, test spliting : done !")

    print("\nCleaning input tweet to predict ...")
    print("\t - before cleaning :", test_tweet)
    print("\t - after cleaning :", normalisation(test_tweet))

    with_cv = []
    without_cv = []

    # Testing KNN
    print("\nInit KNN ...")
    knn_init(x_train, y_train)
    print("\t - init knn : done !")

    print("\nTesting KNN ...")
    score = round(knn_score(x_test, y_test) * 100, 2)
    without_cv.append(score)
    print("\t - knn score = ", score, "%")
    score = round(algo_crossval_score(KNeighborsClassifier(n_neighbors=CONST_BEST_K), X, Y) * 100, 2)
    with_cv.append(score)
    print("\t - knn cross validation score = ", score, "%")
    predict = knn_predict(test_tweet)
    print("\t - predicted [", predict, "]")

    # knn_find_best_k(x_train, y_train, x_test, y_test)

    # Testing MB
    print("\nInit Naive bayes ...")
    nb_init(x_train, y_train)
    print("\t - init naive bayes : done !")

    print("\nTesting NB ...")
    score = round(nb_score(x_test, y_test) * 100, 2)
    without_cv.append(score)
    print("\t - naive bayes score = ", score, "%")
    score = round(algo_crossval_score(MultinomialNB(alpha=CONST_BEST_ALPHA), X, Y) * 100, 2)
    with_cv.append(score)
    print("\t - naive bayes cross validation score = ", score, "%")
    predict = nb_predict(test_tweet)
    print("\t - predicted [", predict, "]")

    # nb_find_best_alpha(x_train, y_train, x_test, y_test)

    # Testing SVM
    print("\nInit SVM ...")
    svm_init(x_train, y_train)
    print("\t - init svm : done !")

    print("\nTesting SVM ...")
    score = round(svm_score(x_test, y_test) * 100, 2)
    without_cv.append(score)
    print("\t - svm score = ", score, "%")
    score = round(algo_crossval_score(SVC(kernel='linear', C=CONST_BEST_C), X, Y) * 100, 2)
    with_cv.append(score)
    print("\t - svm cross validation score = ", score, "%")
    predict = svm_predict(test_tweet)
    print("\t - predicted [", predict, "]")

    show_algorithms_comparison(without_cv, with_cv)
