#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os.path
import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
import string
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

df = pd.read_csv("C:\\Users\\sumad\\OneDrive - San José Unified School District\\Documents\\Fake News sets\\fake_or_real_news.csv")


# In[8]:


df['label'].replace(['REAL', 'FAKE'],
                        [0, 1], inplace=True)
df.head()


# In[9]:


df["title"].fillna('', inplace = True)


# In[10]:


def preprocess_text(text):
    text = ''.join([c for c in text if c not in string.punctuation and c not in string.digits])
    tokens = word_tokenize(text, 'english')
    lemmatiser = WordNetLemmatizer()
    lemmatized = [lemmatiser.lemmatize(word) for word in tokens]
    sw = stopwords.words('english')
    stopped = [word for word in lemmatized if word.lower() not in sw]
    return stopped
preprocess_text(df.loc[0, 'title'])


# In[11]:


X = df['title']
y = df['label']


# In[12]:


X = X.fillna('')
bow_transformer = CountVectorizer(analyzer=preprocess_text).fit(X)
text_bow_train = bow_transformer.transform(X)


# In[13]:


model = RandomForestClassifier()
model.fit(text_bow_train, y)


# In[ ]:





# In[104]:


results = ("REAL", "FAKE")
def single_prediction(title_input):
    userInput = {'title':[title_input]}
    userInput = pd.DataFrame(userInput)
    userInput = userInput['title']
    text_bow_user = bow_transformer.transform(userInput)
    pred = model.predict(text_bow_user)
    return pred[0]

title = input("Enter article title: ")
print(results[single_prediction(title)])


# In[ ]:




