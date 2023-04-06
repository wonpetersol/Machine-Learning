# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt

###############################################################################
# MLP model

tf.random.set_seed(500)

class MLP:
    def __init__(self, embedding_matrix, vocab_size, epochs=20, batch_size=1024, seq_len=30, p = 0.4):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_matrix.shape[-1]
        self.p = p
        self.early_stopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
        self.decay_rate = ReduceLROnPlateau(factor=0.1,min_lr = 0.01,monitor = 'val_loss',verbose = 1)
        self.model = self.build_model()
        # min_lr = 0.01
    
    def build_model(self):
        # input layer
        inputs = tf.keras.Input(shape=(self.seq_len,), dtype="int32", name="input")
        # embeding layer
        embedding_layer = tf.keras.layers.Embedding(self.vocab_size,self.embedding_dim,
                                                  weights=[self.embedding_matrix],
                                                  input_length=self.seq_len,
                                                  trainable=False)
        # embedding layer + dropout
        x = embedding_layer(inputs)
        x = Dropout(self.p)(x)
        # linear layer with relu activation function
        x = Dense(512, activation="relu", name="layer1")(x)
        x = Dense(512, activation="relu", name="layer2")(x)
        # output layer (+ sigmoid to force output between 0 and 1)
        outputs = Dense(1, activation="sigmoid", name="output")(x)
        # construct model
        MLP_model = tf.keras.Model(inputs, outputs)
        print("Model Summary:")
        MLP_model.summary()
        
        return MLP_model
    
    def train(self,x_train, y_train,x_test, y_test):
        # compile model
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        # Train model
        training = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, 
                                validation_data=(x_test, y_test), 
                                callbacks=[self.early_stopping, self.decay_rate])

        # best validation accuracy
        print("Best Validation Accuracy: %s" % (max(training.history['val_accuracy'])))
    
        # plot training and validation accuracy
        plt.plot(training.history['accuracy'])
        plt.plot(training.history['val_accuracy'])
        plt.ylabel('Training Accuracy')
        plt.title('MLP model with GloVe embedding')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        # plot training and validation loss
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.ylabel('Training Loss')
        plt.title('MLP model with GloVe embedding')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        # save model
        self.model.save('MLP.h5')
        
        return training
