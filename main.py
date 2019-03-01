from data_preparation import data_preparation
from Models import Models
from Predict import Predict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
from preprocess import add_article_topic_col
import pickle
warnings.filterwarnings("ignore")

def train_model(new_data):
    #declare objects
    Data_preparation = data_preparation()
    models = Models()
    if new_data:
        #read_data
        data = Data_preparation.read_data_add_labels()
        add_article_topic_col(data)
        data = Data_preparation.add_full_text(data)
        data = Data_preparation.add_binary_topics_col(data)
        data.to_csv('new_data/new_processed_data.csv')
    else:
        data = pd.read_csv('new_data/new_processed_data.csv', index_col=0)

    #for fast debug
    #data = data.sample(n=1000)

    train, test = train_test_split(data, test_size=0.1)
    train1, train2 = train_test_split(train, test_size=0.5)

    #train naive baise model
    nb_model_obj = models.train_NB_model(train1)
    zero_one_train_matrix = Data_preparation.create_zero_one_matrix(nb_model_obj,train2)
    lr_model_obj = models.train_lr_model(zero_one_train_matrix,train2['LABEL'])

    #save model
    if save_model:
        nb_pkl_filename = 'nb_pickle_model.pkl'
        with open(nb_pkl_filename,'wb') as file:
            pickle.dump(nb_model_obj,file)
        lr_pkl_filename = 'lr_pickle_model.pkl'
        with open(lr_pkl_filename,'wb') as file:
            pickle.dump(lr_model_obj,file)

    predict_obj = Predict(nb_model_obj,lr_model_obj)
    nb_prediction = predict_obj.nb_predict(test,data_preparation)
    print('test nb score: ' + str(np.mean(nb_prediction == test['LABEL'])))
    lr_proba, lr_prediction = predict_obj.lr_predict(test,Data_preparation)
    print('test lr score: ' + str(np.mean(lr_prediction == test['LABEL'])))
    predict_obj.get_confusion_matrix(test['LABEL'],lr_prediction,'all')
    quantile_data, quantile_accurate = predict_obj.get_quantile_accurate(test,lr_prediction,lr_proba)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(quantile_accurate)
    #todo add confusion matrix for each band
    for index,row in quantile_accurate.iterrows():
        print(row['probaBand'])
        quantile = quantile_data[quantile_data['probaBand'] == row['probaBand']]
        #predict_obj.get_confusion_matrix(quantile['LABEL'], quantile['prediction'], row['probaBand'])

def test_ui_predict():
    data = pd.read_excel('new_data/approved_538_06-07_2018.xlsx')
    data = data.loc[:50]
    with open('nb_pickle_model.pkl', 'rb') as file:
        nb_model_obj = pickle.load(file)
    with open('lr_pickle_model.pkl', 'rb') as file:
        lr_model_obj = pickle.load(file)
    predict_obj = Predict(nb_model_obj, lr_model_obj)
    data = predict_obj.ui_predcit_nb(data)
    pass







if __name__ == "__main__":
    new_data = False
    save_model = False
    #test_ui_predict()
    train_model(new_data)
