import pandas as pd
import numpy as np
import re
import string

class data_preparation:
    def __init__(self):
        self.topics = None

    def read_data_add_labels(self):
        approved1 = pd.read_excel('new_data/approved_538_06-07_2018.xlsx')
        approved2 = pd.read_excel('new_data/approved_538_07-08_2018.xlsx')
        approved3 = pd.read_excel('new_data/approved_538_08-09_2018.xlsx')
        approved4 = pd.read_excel('new_data/approved_538_09-10_2018.xlsx')
        approved5 = pd.read_excel('new_data/approved_538_10-11_2018.xlsx')
        approved6 = pd.read_excel('new_data/approved_538_11-12_2018.xlsx')
        approved = pd.concat([approved1,approved2,approved3,approved4,approved5,approved6])

        deleted1 = pd.read_excel('new_data/deleted_538_06-07_2018.xlsx')
        deleted2 = pd.read_excel('new_data/deleted_538_07-08_2018.xlsx')
        deleted3 = pd.read_excel('new_data/deleted_538_08-09_2018.xlsx')
        deleted4 = pd.read_excel('new_data/deleted_538_09-10_2018.xlsx')
        deleted5 = pd.read_excel('new_data/deleted_538_10-11_2018.xlsx')
        deleted6 = pd.read_excel('new_data/deleted_538_11-12_2018.xlsx')
        deleted = pd.concat([deleted1,deleted2,deleted3,deleted4,deleted5,deleted6])
        approved['LABEL'] = 1
        deleted['LABEL'] = 0
        data = pd.concat([approved, deleted])
        data.reset_index(inplace=True)
        return data

    def add_if_not_nan(self,hadder, text):
        if pd.isna(text):
            return hadder
        else:
            return hadder + ' ' + text

    def add_full_text(self,data):
        data['full_text'] = data.apply(lambda x: self.add_if_not_nan(x['SUBMIT_TITLE'], x['USER_TEXT']), axis=1)
        return data

    def add_binary_topics_col(self,data,ui_prediction=False):
        if ui_prediction:
            self.topics = ['Unnamed: 16', 'ביקורות ספרים', 'ביקורת אופנה', 'ביקורות מוזיקה', 'מגזין ספרים', 'אמנות',
                           'עוד בעולם התרבות', 'באהבה למוזיקה הישראלית', 'טלוויזיה', 'אולפן המוזיקה', 'קליפר', 'הרצל פינת רפאלי',
                           'ריחו``ל', 'מגזין טלוויזיה', 'קולנוע', 'מגזין קולנוע', 'ספרי ילדים', 'קליפ האנגר', 'ראיונות', 'תווי שי',
                           'תרבות ובידור', 'הגולש האמיץ', 'חדשות מוזיקה', 'רדיו', 'המעצב(נ)ת', 'ספרים', 'פרק ראשון', 'כחול לבן',
                           'ביקורות טלוויזיה', 'ביקורות סרטים', 'אירוויזיון 2019', 'חדשות קולנוע', 'אין סודות בחברה', 'מגזין מוזיקה',
                           'מוזיקה', 'במה', 'מועדון הסרט המופרע', 'חדש על המגש']
        else:
            self.topics = self.get_topics_set(data)
        for topic in self.topics:
            data[topic] = 0
        for index, row in data.iterrows():
            try:
                on_topics = row['TOPIC'].split('|')
                data.loc[index, [*on_topics]] = 1
            except:
                print('no pipe for current row')
        return data



    def get_topics_set(self,data):
        topics_set = set()
        topics_list_unparse = data['TOPIC'].tolist()
        topics_set_unparse = set(topics_list_unparse)
        for topics_with_pipe in topics_set_unparse:
            try:
                topics = topics_with_pipe.split('|')
                for topic in topics:
                    topics_set.add(topic)
            except:
                print('no pipe for topics: ' + str(topics_with_pipe))
        return topics_set


    @staticmethod
    def create_Corpus(data):
        return [row['full_text'] for index, row in data.iterrows()]

    @staticmethod
    def get_only_punctuation(s):
        return re.sub(r'[^{}]+'.format(string.punctuation),'',s)

    def create_zero_one_matrix(self, nb_model_obj, data, ui_prediction = False):
        if self.topics is None:
            self.topics = self.get_topics_set(data)
        if ui_prediction:
            data = data[['full_text', *self.topics]]
        else:
            data = data[['full_text', 'LABEL', *self.topics]]
        zero_one_matrix = pd.DataFrame(index=data.index)
        Corpus = self.create_Corpus(data)
        zero_one_matrix['nb_prediction'] = nb_model_obj.predict(Corpus)
        data['split_text'] = data.apply(lambda x: x['full_text'].split(), axis=1)
        data['length'] = data['split_text'].str.len()
        zero_one_matrix['length_0-2']    = np.where((data['length']<=2  )                         , 1, 0)
        zero_one_matrix['length_2-6']    = np.where((data['length'] > 2 ) & (data['length'] <= 6 ), 1, 0)
        zero_one_matrix['length_6-9']    = np.where((data['length'] > 6 ) & (data['length'] <= 9 ), 1, 0)
        zero_one_matrix['length_9-14']   = np.where((data['length'] > 9 ) & (data['length'] <= 14), 1, 0)
        zero_one_matrix['length_14-20']  = np.where((data['length'] > 14) & (data['length'] <= 20), 1, 0)
        zero_one_matrix['length_20-40']  = np.where((data['length'] > 20) & (data['length'] <= 40), 1, 0)
        zero_one_matrix['length_40-inf'] = np.where((data['length'] > 40)                          ,1, 0)
        data['punctuation_text'] = data.apply(lambda x: self.get_only_punctuation(x['full_text']), axis=1)
        data['punctuation_length'] = data['punctuation_text'].str.len()
        zero_one_matrix['punctuation_0']      = np.where((data['punctuation_length']==0  )                         , 1, 0)
        zero_one_matrix['punctuation_1-2']    = np.where((data['punctuation_length'] > 0 ) & (data['length'] <= 2 ), 1, 0)
        zero_one_matrix['punctuation_2-4']    = np.where((data['punctuation_length'] > 2)  & (data['length'] <= 4)  , 1, 0)
        zero_one_matrix['punctuation_4-8']    = np.where((data['punctuation_length'] > 4)  & (data['length'] <= 8)  , 1, 0)
        zero_one_matrix['punctuation_8-13']   = np.where((data['punctuation_length'] > 8)  & (data['length'] <= 13) , 1, 0)
        zero_one_matrix['punctuation_13-inf'] = np.where((data['punctuation_length'] > 13 )                        , 1, 0)
        zero_one_matrix['have_!!!!'] = np.where('!!!!' in data['full_text'],1,0)
        zero_one_matrix['have_....'] = np.where('....' in data['full_text'], 1, 0)
        #zero_one_matrix['LABEL'] = data['LABEL']
        zero_one_matrix = pd.concat([zero_one_matrix,data[[*self.topics]]],axis=1)
        return zero_one_matrix

