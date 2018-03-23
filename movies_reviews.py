import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

train_path = "aclImdb/train/"  # source data path
test_path = "aclImdb/test/"  # test data path


def txt_to_csv(path, name=""):
    indices = []
    text = []
    rating = []
    i = 0
    for filename in os.listdir(path + "pos"):
        try:
            data = open(path + "pos/" + filename, 'r').read()
            indices.append(i)
            text.append(data)
            rating.append("1")
            i += 1
        except:
            pass

    for filename in os.listdir(path + "neg"):
        try:
            data = open(path + "neg/" + filename, 'r').read()
            indices.append(i)
            text.append(data)
            rating.append("0")
            i += 1
        except:
            pass

    Dataset = list(zip(indices, text, rating))

    df = pd.DataFrame(data=Dataset, columns=['index', 'text', "rating"])
    df.to_csv(name, index=False, header=True)


def stochastic_descent(Xtrain, Ytrain, Xtest, Ytest, param=[]):
    clf = GridSearchCV(SGDClassifier(max_iter=20), param, cv=3)
    print("Fitting // learning from the training data ...")
    clf.fit(Xtrain, Ytrain)
    print("Predicting for the test data ...")
    predict_y = clf.predict(Xtest)

    predict_score = clf.score(Xtest, Ytest)
    return predict_score, clf.best_params_


if __name__ == "__main__":
    txt_to_csv(path=train_path, name="imdb_train.csv")
    txt_to_csv(path=test_path, name="imdb_test.csv")

    # get the training data.
    data = pd.read_csv("imdb_train.csv", header=0)
    Xtrain_text = data['text']
    Ytrain = data['rating']

    # get the training data.
    test_data = pd.read_csv("imdb_test.csv", header=0)
    Xtest_text = test_data['text']
    Ytest = test_data['rating']

    # vectorize the data
    print("vectorize the data . . . ")
    vectorizer = CountVectorizer()
    uni_vectorizer = vectorizer.fit(Xtrain_text)
    Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
    Xtest_uni = uni_vectorizer.transform(Xtest_text)

    # Applying the stochastic descent
    print("Applying the stochastic descent ...")
    penalty = ['l1', 'elasticnet', 'l2']
    loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
            'epsilon_insensitive', 'squared_epsilon_insensitive']
    param_l_p = {'penalty': penalty, 'loss': loss}

    score, best_param = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni, Ytest, param_l_p)
    print("Score: {}   //// for SGD with Param {} ".format(score * 100, best_param))
