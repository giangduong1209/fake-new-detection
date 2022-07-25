import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Dropout
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
import nltk
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import tensorflow as tf
import sklearn
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
nltk.download('punkt')


cols = ['user_name', 'post_message', 'timestamp_post', 'num_like_post', 'num_comment_post', 'num_share_post', 'label']
df = pd.read_csv("./public_train.csv", usecols = cols ,encoding = "utf-8")

# like
for i in range(len(df)):
    if pd.isna(df['num_like_post'][i]):
        df['num_like_post'][i] = 1
    if df['num_like_post'][i] in [' Thế giới đẩy nhanh tốc độ phát triển vaccine ngừa COVID-19 là những tin tức mới nhất trong bản tin Mới nhất dịch COVID-19 của Báo Lao Động ngày 17.3.', ' Kết quả phiên xử đầu tiên cảnh sát Mỹ ghì chết George Floyd... là những tin tức quốc tế đáng chú ý ngày 9.6.', 'unknown', ' Solskjaer hạ knock-out Guardiola', ' Ngang nhiên chiếm dụng đất dự án hồ Yên Sở làm bãi xe tải...là những tin tức kinh tế nóng nhất 24h qua.']:
        df['num_like_post'][i] = 1
    if df['num_like_post'][i] == '54 like':
        df['num_like_post'][i] = 54
for i in range(len(df)):
    df['num_like_post'][i] = int(df['num_like_post'][i])


# comment
for i in range(len(df)):
    if pd.isna(df['num_comment_post'][i]):
        df['num_comment_post'][i] = 1
    if df['num_comment_post'][i] in ['1 comment',' Juventus bỏ xa Inter Milan... là những ảnh chế thú vị nhất 24h giờ qua.', 'unknown']:
        df['num_comment_post'][i] = 1
    if df['num_comment_post'][i] == '10 comment':
        df['num_comment_post'][i] = 10
    if df['num_comment_post'][i] == '12 comment':
        df['num_comment_post'][i] = 12
for i in range(len(df)):
    df['num_comment_post'][i] = int(df['num_comment_post'][i])

# share
for i in range(len(df)):
    if pd.isna(df['num_share_post'][i]):
        df['num_share_post'][i] = 1
    if df['num_share_post'][i] in ['May 25th 2020, 21:57:58.000','June 11th 2020, 19:03:45.000','1 share','unknown']:
        df['num_share_post'][i] = 1
for i in range(len(df)):
    df['num_share_post'][i] = float(df['num_share_post'][i])
for i in range(len(df)):
    df['num_share_post'][i] = int(df['num_share_post'][i])

numerical_cols = ['num_like_post',	'num_comment_post',	'num_share_post']
for col in numerical_cols:
    scale = StandardScaler().fit(df[[col]])
    df[col] = scale.transform(df[[col]])  
    df[col] = scale.transform(df[[col]])

df["post_message"] = df["post_message"].str.lower() # Chuyển text về lowercasse
df = df.dropna()


embedding_dict = {}
with open('./vocab_phoBert.txt', 'r', encoding='utf-8') as phobert:
    for line in phobert:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1])
        embedding_dict[word] = vectors
phobert.close()


def create_corpus(df):
    corpus = []
    for tweet in tqdm(df['post_message']):
        words = [word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)

    return corpus

df['post_message']=df['post_message'].apply(str)
corpus = create_corpus(df)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_matrix(corpus, mode='tfidf')
tweet_pad = pad_sequences(sequences, maxlen=200, truncating='post', padding='post')

word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, 200))

for word, i in tqdm(word_index.items()):
    if i > num_words:
        continue
        
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from keras import regularizers

model = Sequential()

phobert_embedding = Embedding(num_words, 200, embeddings_initializer = Constant(embedding_matrix), 
                     input_length = 200, 
                     trainable = False)

import keras
'''
model = Sequential()
#Embedding layer
model.add(phobert_embedding)
model.add(LSTM(units = 128, activation = 'tanh', recurrent_activation = 'sigmoid', return_sequences = True, recurrent_dropout = 0, dropout = 0.25 ))
model.add(LSTM(units = 64,activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, dropout = 0.1))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
'''
model = keras.Sequential([
    phobert_embedding,
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()


from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(tweet_pad, df['label'], train_size= 0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)

history = model.fit(X_train, y_train, 
                    epochs=80, verbose=1, 
                    validation_data=(X_valid, y_valid))


model.save("model.h5")


