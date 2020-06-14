import numpy as np
import pandas as pd
import operator
from wordcloud import STOPWORDS
import string

pd.set_option('display.max_rows', 5000000)
pd.set_option('display.max_columns', 5000000)
pd.set_option('display.width', 10000000)
pd.set_option('display.max_colwidth', 4000)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv') 

import re
def clean(tweet): 
    
    # Special characters
   
   
    # Urls
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
        
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        tweet = tweet.replace(p, f' {p} ')
        
    # ... and ..
    tweet = tweet.replace('...', ' ... ')
    if '...' not in tweet:
        tweet = tweet.replace('..', ' ... ')      
        
    # Acronyms
    tweet = re.sub(r"MH370", "Malaysia Airlines Flight 370", tweet)
    tweet = re.sub(r"mÌ¼sica", "music", tweet)
    tweet = re.sub(r"okwx", "Oklahoma City Weather", tweet)
    tweet = re.sub(r"arwx", "Arkansas Weather", tweet)    
    tweet = re.sub(r"gawx", "Georgia Weather", tweet)  
    tweet = re.sub(r"scwx", "South Carolina Weather", tweet)  
    tweet = re.sub(r"cawx", "California Weather", tweet)
    tweet = re.sub(r"tnwx", "Tennessee Weather", tweet)
    tweet = re.sub(r"azwx", "Arizona Weather", tweet)  
    tweet = re.sub(r"alwx", "Alabama Weather", tweet)
    tweet = re.sub(r"wordpressdotcom", "wordpress", tweet)    
    tweet = re.sub(r"usNWSgov", "United States National Weather Service", tweet)
    tweet = re.sub(r"Suruc", "Sanliurfa", tweet)   
    
    # Grouping same words without embeddings
    tweet = re.sub(r"Bestnaijamade", "bestnaijamade", tweet)
    tweet = re.sub(r"SOUDELOR", "Soudelor", tweet)
    
    return tweet


df_train['text_cleaned'] = df_train['text'].apply(lambda s : clean(s))
df_test['text_cleaned'] = df_test['text'].apply(lambda s : clean(s))



df_mislabeled = df_train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
df_mislabeled.index.tolist()

df_train['target_relabeled'] = df_train['target'].copy() 

df_train.loc[df_train['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'To fight bioterrorism sir.', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!', 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG', 'target_relabeled'] = 1
df_train.loc[df_train['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "Caution: breathing may be hazardous to your health.", 'target_relabeled'] = 1
df_train.loc[df_train['text'] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0

# word_count
df_train['word_count'] = df_train['text'].apply(lambda x: len(str(x).split()))
df_test['word_count'] = df_test['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df_train['unique_word_count'] = df_train['text'].apply(lambda x: len(set(str(x).split())))
df_test['unique_word_count'] = df_test['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
df_train['stop_word_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
df_test['stop_word_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
df_train['url_count'] = df_train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))
df_test['url_count'] = df_test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
df_train['mean_word_length'] = df_train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df_test['mean_word_length'] = df_test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df_train['char_count'] = df_train['text'].apply(lambda x: len(str(x)))
df_test['char_count'] = df_test['text'].apply(lambda x: len(str(x)))

# punctuation_count
df_train['punctuation_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df_test['punctuation_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
df_train['hashtag_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
df_test['hashtag_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
df_train['mention_count'] = df_train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
df_test['mention_count'] = df_test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split( df_train[["text",'text_cleaned', "word_count", "unique_word_count","stop_word_count","mean_word_length","char_count","punctuation_count"]]
                                                     , df_train["target_relabeled"], test_size=0.2,
                                                       random_state = 1)

train_x1 = train_x["text_cleaned"]
train_x2 = train_x["word_count"]
train_x3 = train_x["unique_word_count"]
train_x4 = train_x["stop_word_count"]
train_x5 = train_x["mean_word_length"]
train_x6 = train_x["char_count"]
train_x7 = train_x["punctuation_count"]

test_x1 = test_x["text_cleaned"]
test_x2 = test_x["word_count"]
test_x3 = test_x["unique_word_count"]
test_x4 = test_x["stop_word_count"]
test_x5 = test_x["mean_word_length"]
test_x6 = test_x["char_count"]
test_x7 = test_x["punctuation_count"]

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit_transform(train_x1)

x_train_tf_idf_word1 = tf_idf_word_vectorizer.transform(train_x1)

from scipy.sparse import coo_matrix, hstack

x_train_tf_idf_word2 = coo_matrix(train_x2)
x_train_tf_idf_word2 = x_train_tf_idf_word2.reshape(6090,1)

x_train_tf_idf_word3 = coo_matrix(train_x3)
x_train_tf_idf_word3 = x_train_tf_idf_word3.reshape(6090,1)

x_train_tf_idf_word4 = coo_matrix(train_x4)
x_train_tf_idf_word4 = x_train_tf_idf_word4.reshape(6090,1)

x_train_tf_idf_word5 = coo_matrix(train_x5)
x_train_tf_idf_word5 = x_train_tf_idf_word5.reshape(6090,1)

x_train_tf_idf_word6 = coo_matrix(train_x6)
x_train_tf_idf_word6 = x_train_tf_idf_word6.reshape(6090,1)

x_train_tf_idf_word7 = coo_matrix(train_x7)
x_train_tf_idf_word7 = x_train_tf_idf_word7.reshape(6090,1)

x_train_tf_idf_word = hstack([x_train_tf_idf_word1,x_train_tf_idf_word2,x_train_tf_idf_word3, x_train_tf_idf_word4, x_train_tf_idf_word5, x_train_tf_idf_word6, x_train_tf_idf_word7])

x_train_tf_idf_word = x_train_tf_idf_word.toarray()

x_test_tf_idf_word1 = tf_idf_word_vectorizer.transform(test_x1)

x_test_tf_idf_word2 = coo_matrix(test_x2)
x_test_tf_idf_word2 = x_test_tf_idf_word2.reshape(1523,1)

x_test_tf_idf_word3 = coo_matrix(test_x3)
x_test_tf_idf_word3 = x_test_tf_idf_word3.reshape(1523,1)

x_test_tf_idf_word4 = coo_matrix(test_x4)
x_test_tf_idf_word4 = x_test_tf_idf_word4.reshape(1523,1)

x_test_tf_idf_word5 = coo_matrix(test_x5)
x_test_tf_idf_word5 = x_test_tf_idf_word5.reshape(1523,1)

x_test_tf_idf_word6 = coo_matrix(test_x6)
x_test_tf_idf_word6 = x_test_tf_idf_word6.reshape(1523,1)

x_test_tf_idf_word7 = coo_matrix(test_x7)
x_test_tf_idf_word7 = x_test_tf_idf_word7.reshape(1523,1)

x_test_tf_idf_word = hstack([x_test_tf_idf_word1, x_test_tf_idf_word2,x_test_tf_idf_word3,x_test_tf_idf_word4,x_test_tf_idf_word5,x_test_tf_idf_word6,x_test_tf_idf_word7])

x_test_tf_idf_word = x_test_tf_idf_word.toarray()

test1 = df_test["text_cleaned"]
test1 = tf_idf_word_vectorizer.transform(test1)

test2 = df_test["word_count"]
test2 = coo_matrix(test2)
test2 = test2.reshape(3263,1)

test3 = df_test["unique_word_count"]
test3 = coo_matrix(test3)
test3 = test3.reshape(3263,1)

test4 = df_test["stop_word_count"]
test4 = coo_matrix(test4)
test4 = test4.reshape(3263,1)

test5 = df_test["mean_word_length"]
test5 = coo_matrix(test5)
test5 = test5.reshape(3263,1)

test6 = df_test["char_count"]
test6 = coo_matrix(test6)
test6 = test6.reshape(3263,1)

test7 = df_test["punctuation_count"]
test7 = coo_matrix(test7)
test7 = test7.reshape(3263,1)

test = hstack([test1, test2,test3,test4,test5,test6,test7])

test = test.toarray()

from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np


from keras.models import Sequential

# Input - Layer

#BU YENİ MODEL
from keras.layers import LeakyReLU
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM



#model = Sequential()
#model.add(Dense(128, kernel_initializer ='glorot_uniform',input_dim=XX.shape[1]))
#model.add(LeakyReLU(alpha=0.01))
#model.add(Dropout(0.20))
#model.add(Dense(128, kernel_initializer ='glorot_uniform'))
#model.add(LeakyReLU(alpha=0.01))
#model.add(Dropout(0.20))
#model.add(Dense(output_dim = 1, kernel_initializer ='glorot_uniform', activation = 'sigmoid'))
#model.compile(loss='binary_crossentropy',
#              optimizer='adamax',
#              metrics=['acc',f1_m,precision_m, recall_m])
#
#es = keras.callbacks.EarlyStopping(monitor='val_f1_m', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#
#model.summary()
#
#model.fit(XX, Y, batch_size = 12, nb_epoch = 100,callbacks=[es],validation_split=0.2)



#BU KISIM ORJİNAL MODEL 
model = Sequential()
model.add(layers.Dense(10, activation = "relu", input_shape=(14847, )))
# Hidden - Layers

model.add(layers.Dense(5, activation = "l_relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit(
 x_train_tf_idf_word, train_y,
 epochs= 11,
 batch_size = 150,
 validation_data = (x_test_tf_idf_word, test_y)
)

y_pred_test = model.predict(x_test_tf_idf_word)

y_pred_test = model.predict(x_test_tf_idf_word)
y_pred_test = (y_pred_test > 0.5)
y_pred_test = pd.DataFrame(data = y_pred_test, index = range(1523), columns=['target'])
y_pred_test["target"] = y_pred_test["target"].astype(int)

y_pred = model.predict(test)
y_pred = (y_pred > 0.5)
y_pred1 = pd.DataFrame(data = y_pred, index = range(3263), columns=['target'])
y_pred1["target"] = y_pred1["target"].astype(int)

from sklearn.metrics import classification_report,confusion_matrix
cm = confusion_matrix(test_y, y_pred_test)
print(cm)

test_x1 = test_x1.reset_index()
test_y = test_y.reset_index()
veri_inceleme_test = pd.concat([test_x1["text_cleaned"], y_pred_test, test_y], axis=1)

veri_inceleme_test.drop(["index"],axis=1,inplace=True)    ###Drop Sütun Etmek

