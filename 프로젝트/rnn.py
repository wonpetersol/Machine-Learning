from typing import final
import numpy as np
import pandas as pd
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, Activation, Embedding, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical

from keras import Input
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

seed = 500

def clean(df):
    # remove extra chars
    # returns another dataframe with cleaned text
    tweets = df['tweet']
    blacklist = ['@', '#', 'http', 'www', '/']
    cleaned_tweets = []
    for t in tweets:
        text = t.split()
        cleaned = [e for e in text if all(ch not in e for ch in blacklist)]
        cleaned_text = ' '.join(cleaned)
        cleaned_tweets.append(cleaned_text)
    df['tweet'] = cleaned_tweets
    return df

def tokenize_tweets(df):
    tweets = df['tweet']
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = []
    for i in tweets:
        tokens.append(tokenizer.tokenize(i.lower()))
    return tokens

def remove_stopwords(tokenized):
    # sw = set(stopwords.words('english'))
    sw = {'ll', 'did', 'those', 'during', 'won', 'on', "that'll", 've', 'they', 'hers', 'once', 'is', 'here', 
    'off', 'over', 'each', 'y', 'nor', 'some', 'theirs', 'into', 'other', 'what', 
    'then', 'as', 'both', 'herself', 'why', 'after', 'because', 'its', 'under', 'which', 'we', 
    'all', 'yourself', 'from', 'the', 'didn', 'just', 'most', 'in', 'so', 'does', 'have', 'myself', 
    'out', 'o', 'having', 'him', 'there', 'that', 'if', 'to', 'any', 'above', 'for', "you're", 'itself', 'by', 'ma', 
    'me', 'd', 'while', 'than', 'very', 'up', 'this', 'm', 'between', 'doing', 'few', 
    'more', 'until', 're', 'i', 'who', 'against', 'are', 'an', 'our', 'further', 'has', 'your', 'down', 'of', 'you', 
    'my', 'been', 'before', 'yourselves', 'how', 'himself', 'such', 'can', 'will', 'his', 'but', 'them', 'only', 
    "she's", 'same', "it's", 'a', 'do', 's', 'am', 'below', 'now', 'her', 'should', "you've", 'it', 'hasn', 
    'or', 'were', 'being', 'she', 'ours', 'had', 'about', 'be', 'and', "you'll", 'too', 'yours', 
    'themselves', 'whom', "should've", 'he', 'these', 'again', 'ourselves', 'at', 'own', 'with', 'where', 
    'their', 'was', "you'd", 'when', 'through'}
    sw_removed = []
    for sent in tokenized:
        filtered = []
        for word in sent:
            if word not in sw:
                filtered.append(word)
        sw_removed.append(filtered)
    return sw_removed

def lemmatize(processed):
    lemma = WordNetLemmatizer()
    lemmatized = []
    for sent in processed:
        filtered = []
        for word in sent:
            filtered.append(lemma.lemmatize(word, pos='v'))
        lemmatized.append(filtered)
    return lemmatized

def sample(df):
    sampled = df.groupby(0).apply(lambda x: x.sample(160000))
    sampled.index = sampled.index.droplevel(0)
    return sampled

def prepare_dataset():
    # returns prepared dataset with cleaned tweets
    raw = pd.read_csv('data.csv', usecols=[0,5], header=None, encoding='latin-1')
    
    df = sample(raw).reset_index(drop=True)
    # print(df.head(15))
    df.rename(columns={0:'y', 5:'tweet'}, inplace=True)
    df['y'] = df['y'].replace(4, 1)
    cleaned_df = clean(df)
    # print(cleaned_df.head(10))
    # tokenized, sw_removed, lemmatized are 2d lists
    tokenized = tokenize_tweets(cleaned_df)
    # sw_removed = remove_stopwords(tokenized)
    # lemmatized = lemmatize(sw_removed)
    lemmatized = lemmatize(tokenized)
    # merge into df for ease of use
    final_tweets = []
    for entry in lemmatized:
        final_tweets.append(' '.join(entry))
    d = {'y':df['y'], 'tweets':final_tweets}
    final_df = pd.DataFrame(data=d)
    #print(final_df.head(10))
    return final_df

def prepare_glove(word_indices, vocab_size, embedding_dim):
    glove = {}
    glove_matrix = np.zeros((vocab_size, embedding_dim))
    with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            l = line.split()
            word = l[0]
            vals = np.array(l[1:], dtype=np.float32)
            glove[word] = vals

    for word, ind in word_indices.items():
        vector = glove.get(word)
        if vector is not None:
            glove_matrix[ind] = vector
    return glove_matrix

def train_test_split(df, test_size=0.2, seq_len=30, embedding_dim=100):
    # returns split data and vocab size
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df['tweets'], df['y'], test_size=test_size, random_state=seed)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['tweets'])
    word_indices = tokenizer.word_index
    vocab_size = len(word_indices) + 1
    
    #print(vocab_size)
    #print(seq_len)
    #print(embedding_dim)
    
    glove_matrix = prepare_glove(word_indices, vocab_size, embedding_dim)
    
    # pad sequences to same length
    x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=seq_len)
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=seq_len)
    print("glove_matrix shape: ", glove_matrix.shape)
    print("x train: ", x_train.shape)
    print("x test: ", x_test.shape)

    return x_train, x_test, y_train, y_test, vocab_size, glove_matrix

#main
#nltk.download('wordnet')
final_df = prepare_dataset()
x, xt, y, yt, _, _ =  train_test_split(final_df)

scaler = MinMaxScaler()

x_data = np.concatenate((x,xt))
#x_data = scaler.fit_transform(x_data)
x_data = np.reshape(x_data,(x_data.shape[0],x_data.shape[1],1))

x_train = np.reshape(x,(x.shape[0],x.shape[1],1))
x_test = np.reshape(xt,(xt.shape[0],xt.shape[1],1))

y_data = pd.Series(np.concatenate((y,yt)))
y_data = np.reshape(y_data.values,(len(y_data),1))

y_train = np.reshape(y.values,(len(y),1))
y_test = np.reshape(yt.values,(len(yt),1))


#vanillaRNN Model
model = Sequential()
model.add(SimpleRNN(4, input_shape=(30,1),return_sequences=True))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#adam = optimizers.Adam(learning_rate=0.01)
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
history = model.fit(x_data, y_data, validation_split=0.2, epochs=20, batch_size=512)

#history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#LSTM Model
model = Sequential()
model.add(LSTM(4, input_shape=(30,1),return_sequences=True))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#adam = optimizers.Adam(learning_rate=0.01)
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

history = model.fit(x_data, y_data, validation_split=0.2, epochs=20, batch_size=512)

#history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#LSTM Hyperparameter Tuned
model = Sequential()
#model.add(Embedding(96961, 100, input_length=30))
model.add(Embedding(97500, 100, input_length=30))
model.add(Dropout(0.2)) #0.2
model.add(LSTM(4, input_shape=(30,1),return_sequences=True))
model.add(Dropout(0.2)) #0.2
model.add(LSTM(4, return_sequences=True))
model.add(Dropout(0.2)) #0.2
model.add(LSTM(4))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = optimizers.Adam(learning_rate=0.01)
sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #0.01
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(x_data, y_data, validation_split=0.1, epochs=20, batch_size=500)

#history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()
