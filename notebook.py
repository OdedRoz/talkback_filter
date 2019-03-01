
# coding: utf-8

# In[196]:


import pandas as pd
from collections import Counter
import os
import re
cwd = os.getcwd()
cwd


# In[234]:


def add_if_not_nan(hadder,text):
    if pd.isna(text):
        return hadder
    else:
        return hadder + ' ' + text

def count_common_words(data):
    c = Counter()
    for index, row in data.iterrows():
        c.update(row['split_text'])
    return c

def get_only_punctuation(s):
    return re.sub(r'[^{}]+'.format(punctuation),'',s)

def my_punctuation_band(i):
    if i == 0:
        return '0'
    elif (i>=1 and i<=2):
        return '1-2'
    elif (i>=3 and i<=4):
        return '3-4'
    elif (i>=5 and i<=8):
        return '4-8'
    elif (i>=9 and i<=12):
        return '9-12'
    elif (i>=13 and i<=20):
        return '13-20'
    else:
        return '>20'


# In[161]:


approved = pd.read_csv(r'.\Documents\Technion\final_project\approved_04-05_2018.csv')
deleted = pd.read_csv(r'.\Documents\Technion\final_project\deleted_04-05_2018.csv')


# In[71]:


#for debug
approved = approved[:100]
deleted = deleted[:100]


# In[159]:


len(approved)


# In[163]:


len(deleted)


# In[162]:


#make data blanaced
deleted = deleted.sample(66238)


# In[44]:


approved.head()


# In[45]:


deleted.head()


# In[164]:


approved['LABEL']=1
deleted['LABEL']=-1


# In[165]:


data = pd.concat([approved,deleted])


# In[166]:


data['full_text'] = data.apply(lambda x: add_if_not_nan(x['SUBMIT_TITLE'],x['USER_TEXT']), axis = 1)


# In[167]:


data = data[['full_text','LABEL']]


# In[168]:


data['split_text'] = data.apply(lambda x: x['full_text'].split(),axis = 1) 


# In[169]:


aproved_common_words = count_common_words(data.loc[data['LABEL'] == 1])
deleted_common_words = count_common_words(data.loc[data['LABEL'] == -1])


# In[170]:


aproved_common_words.most_common()


# In[171]:


deleted_common_words.most_common()


# In[172]:


data['length'] = data['split_text'].str.len()


# In[177]:


length_data = data[data['length']<=100]
hist = length_data[['length']].hist(bins=100)
hist


# In[178]:


length_data = data[data['length']<=20]
hist = length_data[['length']].hist(bins=20)
hist


# In[173]:


data['lengthBand'] = pd.qcut(data['length'], 8)


# In[174]:


data.groupby(['lengthBand']).size()


# In[175]:


data[['lengthBand','LABEL']].groupby(['lengthBand'], as_index=False).mean()


# In[200]:


data['punctuation_text'] = data.apply(lambda x: get_only_punctuation(x['full_text']), axis = 1)


# In[206]:


data['punctuation_length'] = data['punctuation_text'].str.len()


# In[209]:


punctuation_data = data[data['punctuation_length']<=100]
hist = punctuation_data[['punctuation_length']].hist(bins=100)
hist


# In[210]:


punctuation_data = data[data['punctuation_length']<=20]
hist = punctuation_data[['punctuation_length']].hist(bins=20)
hist


# In[228]:


data['punctuationBand'] = pd.qcut(data['punctuation_length'],5,duplicates='drop')


# In[231]:


data.groupby(['punctuationBand']).size()


# In[232]:


data[['punctuationBand','LABEL']].groupby(['punctuationBand'], as_index=False).mean()


# In[235]:


data['my_punctuation_band'] = data.apply(lambda x: my_punctuation_band(x['punctuation_length']), axis = 1)


# In[236]:


data.groupby(['my_punctuation_band']).size()


# In[238]:


data[['my_punctuation_band','LABEL']].groupby(['my_punctuation_band'], as_index=False).mean()


# In[239]:


data[data['my_punctuation_band']=='>20']

