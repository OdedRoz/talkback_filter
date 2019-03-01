import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

approved_path = 'C:\Final Project\\approved.csv'
deleted_path = 'C:\Final Project\deleted.csv'


def add_article_topic_col(df):
    df['TOPIC'] = ''
    topic_dict = {}
    for index, row in df.iterrows():
        if row['ARTICLE_ID'] in topic_dict.keys():
            df['TOPIC'][index] = topic_dict[row['ARTICLE_ID']]
        else:
            try:
                topic = get_article_topic(row['ARTICLE_ID'])
                df['TOPIC'][index] = topic
                topic_dict[row['ARTICLE_ID']] = topic
            except:
                continue


        if index % 100 == 0:
            print(index, 'rows were processed')


def get_article_topic(article_id):
    url_path = 'https://www.ynet.co.il/articles/0,7340,L-{},00.html'.format(int(article_id))
    r = requests.get(url_path)
    soup = BeautifulSoup(r.text)
    topic = ''
    flag = False
    for a_tag in soup.find_all('a'):
        if a_tag.text and 'name' in a_tag.attrs and a_tag.attrs['name'] == "top":
            topic += a_tag.text + '|'
            flag = True
        elif flag:
            break
    return topic[:-1]


def split_to_train_and_test(df):
    train = df.sample(frac=0.85)
    test = df.drop(train.index)
    return train, test


def merge_approved_and_deleted(approved, deleted):
    return pd.concat([approved,deleted]).reset_index(drop=True)


def load_from_csv(csv_path):
    return pd.read_csv(csv_path, encoding='ansi')


def add_sentiment_col(df, add_ones):
    df_len = df.__len__()
    if add_ones:
        return df.assign(SENTIMENT=pd.Series(np.ones(df_len)))
    else:
        return df.assign(SENTIMENT=pd.Series(np.zeros(df_len)))


def concat_text(df):
    df['TEXT'] = df.TOPIC + ' ' + df.SUBMIT_TITLE.fillna('') + ' ' + df.USER_TEXT.fillna('')
    return df


def main():

    '''
    chars = set('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM')
    del_ind = []
    for index, row in data.iterrows():
        if any((c in chars) for c in row['TEXT']):
            if 'ynet' not in row['TEXT'] and 'Ynet' not in row['TEXT']:
                del_ind.append(index)
            print('Found english letter in ind: ', index, '| Value: ', row['TEXT'])
    print('Found ', len(del_ind), ' Rows')
    en_free_data = data.drop(del_ind)
    #en_free_data = clean_english(data)
    print('after shape', en_free_data.shape)
    en_free_data.to_csv('C:\Final Project\\en_free_data_2.csv')
    '''

    df_approved = load_from_csv(approved_path)
    add_article_topic_col(df_approved)
    df_approved = concat_text(add_sentiment_col(df_approved, add_ones=True))
    df_approved = df_approved[['TEXT', 'SENTIMENT']]
    
    df_deleted = load_from_csv(deleted_path)
    add_article_topic_col(df_deleted)
    df_deleted = concat_text(add_sentiment_col(df_deleted, add_ones=False))
    df_deleted = df_deleted[['TEXT', 'SENTIMENT']]

    df_merged = merge_approved_and_deleted(df_approved, df_deleted)
    df_merged.to_csv('C:\Final Project\data.csv')


if __name__ == "__main__":
    main()