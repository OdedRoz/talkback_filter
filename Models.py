import pandas as pd
from data_preparation import data_preparation
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np



class Models:
    def __init__(self):
        self.data_preparation = data_preparation()

    def train_NB_model(self, train):
        trainCorpus = self.data_preparation.create_Corpus(train)
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        text_clf.fit(trainCorpus, train['LABEL'])
        trainPredicted = text_clf.predict(trainCorpus)
        print('train score: ' + str(np.mean(trainPredicted == train['LABEL'])))
        return text_clf

    def train_lr_model(self, X,Y):
        lr_clf = LogisticRegression().fit(X,Y)
        trainPredicted = lr_clf.predict(X)
        print('train score: ' + str(np.mean(trainPredicted == Y)))
        return lr_clf





