
import tornado.ioloop
import tornado.web
from tornado.httpclient import HTTPClient

from main import *

print("CONFIGURING DATASET")
X, Y = create_set(dataset_raw_path)
X = encode_set(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
init_all(x_train, y_train)
print("DONE")


knn = round(knn_score(x_test, y_test), 4) * 100
knn_cv = round(algo_crossval_score(KNeighborsClassifier(n_neighbors=CONST_BEST_K), X, Y) * 100, 2) 
print("KNN DONE")
nb = round(nb_score(x_test, y_test), 4) * 100
nb_cv = round(algo_crossval_score(MultinomialNB(alpha=CONST_BEST_ALPHA), X, Y) * 100, 2)
print("NB DONE")
svm = round(svm_score(x_test, y_test), 4) * 100
svm_cv = round(algo_crossval_score(SVC(kernel='linear', C=CONST_BEST_C), X, Y) * 100, 2)
print("SVM DONE")


class StartHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("res/index.html")

    def post(self):
        dest = self.get_argument('name')
        if dest == "metrics" :
            self.redirect("/fdd/mettrics")
            return
        if dest == "predict":
            self.redirect("/fdd/predict")
            return

class PredictHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("res/predict.html")

    def post(self):
        tweet = self.get_argument('tweet')
        predict_nb = nb_predict(tweet)
        predict_knn = knn_predict(tweet)
        predict_svm = svm_predict(tweet)
        result = knn_predict(tweet)
        self.render("res/output.html", tweet=tweet, predict_nb=predict_nb, predict_knn=predict_knn, predict_svm=predict_svm)


class MetricsHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("res/metrics.html", knn=knn, knn_cv=knn_cv, nb=nb, nb_cv=nb_cv, svm=svm, svm_cv=svm_cv)

    def post(self):
        self.redirect("/fdd")
        return

def make_app():
    return tornado.web.Application([
        (r"/fdd", StartHandler),
        (r"/fdd/predict", PredictHandler),
        (r"/fdd/metrics", MetricsHandler),
        
    ])

if __name__ == "__main__":
    print("LISTENING")
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()