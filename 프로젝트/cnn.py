import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D, Dropout, Embedding, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.random.set_seed(500)

class CNN:
    def __init__(self, vocab_size, glove_matrix, batch_size=512, epochs=20, embedding_dim=100, seq_len=30):
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=0.000001, verbose=0)
        self.model = self.build_model(vocab_size, glove_matrix)

    def build_model(self, vocab_size, glove_matrix):
        inputs = tf.keras.Input(shape=(self.seq_len,), dtype=np.int32)
        embedding_layer = Embedding(vocab_size, self.embedding_dim, weights=[glove_matrix], input_length=self.seq_len, trainable=False)
        model = embedding_layer(inputs)
        model = Dropout(0.2)(model)

        # conv block 1
        model = Conv1D(128, 3, activation='relu')(model) #128
        model = Conv1D(64, 3, activation='relu')(model) #64
        model = MaxPooling1D()(model)

        # conv block 2
        model = Conv1D(64, 3, activation='relu')(model) #64
        model = Conv1D(32, 3, activation='relu')(model) #32
        model = Conv1D(32, 3, activation='relu')(model) #32
        model = Conv1D(32, 3, activation='relu')(model) #16
        
        # conv block 3
        # model = Conv1D(32, 3, activation='relu')(model)
        # model = Conv1D(32, 3, activation='relu')(model)
        # model = Conv1D(16, 3, activation='relu')(model)
        model = GlobalMaxPooling1D()(model)

        # FC
        model = Dense(1024, activation='relu')(model)
        model = Dense(512, activation='relu')(model)
        model = Dropout(0.2)(model)

        # output layer
        outputs = Dense(1, activation='sigmoid')(model)
        return tf.keras.Model(inputs, outputs)

    def summary(self):
        return self.model.summary()

    def train(self, xtrain, xtest, ytrain, ytest, plot=False, save=False, callbacks=False):
        opt = Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        if callbacks:
            history = self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.epochs, validation_data=(xtest, ytest), 
                callbacks=[self.early_stop, self.reduce_lr])
        else:
            history = self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.epochs, validation_data=(xtest, ytest))

        # plot
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Training Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('CNN_Acc.png')
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('CNN_Loss.png')
        plt.clf()

        # save model
        if save:
            self.model.save('models/CNN.h5')