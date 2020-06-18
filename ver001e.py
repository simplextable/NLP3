#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


# In[3]:


import os
#os.listdir('C:\Users\Predator\Desktop\Pyton ile Makine Öğrenmesi\glove.840B.300d.pkl')


# In[2]:


tweet= pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
tweet.head(3)


# In[3]:


df=pd.concat([tweet,test])


# In[4]:


df.head()


# In[5]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


# In[6]:


df['text']=df['text'].apply(lambda x : remove_URL(x))


# In[7]:


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# In[8]:


df['text']=df['text'].apply(lambda x : remove_html(x))


# In[9]:


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# In[10]:


df['text']=df['text'].apply(lambda x: remove_emoji(x))


# In[11]:


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


# In[12]:


df['text']=df['text'].apply(lambda x : remove_punct(x))


# In[19]:


from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
        
text = "corect me plese"
correct_spellings(text)


# In[13]:


import nltk
nltk.download('punkt')


# In[14]:


def create_corpus(df):
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus


# elimizdeki df verisi satır satır ve kelime kelime ayrıldı.

# In[15]:


corpus=create_corpus(df)


# In[16]:


print(corpus)


# In[18]:



# In[22]:


embedding_dict={}
with open('C:\\Users\\Predator\\Desktop\\Pyton ile Makine Öğrenmesi\\glove.6B.100d.txt','r',encoding="utf8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
    f.close()


# In[24]:


MAX_LEN=50
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences=tokenizer_obj.texts_to_sequences(corpus)

tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')


# In[50]:


tweet_pad


# In[35]:


word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))


# In[38]:


print(len(word_index))


# In[36]:


num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec


# In[45]:


np.shape(embedding_matrix)


# In[47]:


model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])


# In[51]:


train=tweet_pad[:tweet.shape[0]]
test=tweet_pad[tweet.shape[0]:]


# In[52]:


X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)


# In[53]:


history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)


# In[55]:


predictions = model.predict_classes(X_test)


# In[58]:


from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[60]:


plot_cm(y_test,predictions, 'Confution matrix of Tweets', figsize=(5,5))


# In[62]:


submission=pd.read_csv('sample_submission.csv')
y_pre=model.predict(test)
y_pre=np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':submission['id'].values.tolist(),'target':y_pre})
sub.to_csv('submission.csv',index=False)


# In[ ]:

y_pre1=model.predict(X_test)



