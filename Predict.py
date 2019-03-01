import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from data_preparation import data_preparation
from preprocess import add_article_topic_col

class Predict:
    def __init__(self,nb_model_obj,lr_model_obj):
        self.nb_model_obj = nb_model_obj
        self.lr_model_obj = lr_model_obj


    def lr_predict(self, data, data_preparation, ui_prediction = False):
        zero_one_matrix = data_preparation.create_zero_one_matrix(self.nb_model_obj, data, ui_prediction)
        proba = self.lr_model_obj.predict_proba(zero_one_matrix)
        prediction = self.lr_model_obj.predict(zero_one_matrix)
        return proba,prediction

    def nb_predict(self, data,data_preparation):
        Corpus = data_preparation.create_Corpus(data)
        return self.nb_model_obj.predict(Corpus)

    def get_quantile_accurate(self, data, prediction, proba):
        data['prediction'] = prediction
        data['accurate'] = np.where((data['prediction'] == data['LABEL']), 1, 0)
        data['0_proba'] = proba[:, 0]
        data['1_proba'] = proba[:, 1]
        data['proba'] = data[['0_proba', '1_proba']].max(axis=1)
        data['probaBand'] = pd.qcut(data['proba'], 10, duplicates='drop')
        return data, data[['probaBand', 'accurate']].groupby(['probaBand'], as_index=False).mean()

    def get_confusion_matrix(self,y_real,y_predicted,data_part):
        cnf_matrix = confusion_matrix(y_real, y_predicted)
        np.set_printoptions(precision=2)
        class_names = ['0','1']
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix '+str(data_part))
        plt.show()

    def ui_predict(self, data):
        data_perp = data_preparation()
        add_article_topic_col(data)
        data = data_perp.add_full_text(data)
        data = data_perp.add_binary_topics_col(data,ui_prediction=True)
        proba, prediction = self.lr_predict(data, data_perp, ui_prediction=True)


    def ui_predcit_nb(self, data):
        data_perp = data_preparation()
        data = data_perp.add_full_text(data)
        Corpus = data_perp.create_Corpus (data)
        data['Prediction'] = self.nb_model_obj.predict_proba(Corpus)[:,1]
        return data




    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()






