import pandas as pd
import dataset
import matplotlib.pyplot as plt
import tensorflow.keras.models as tfk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import sys

def load_model(path):
    return tfk.load_model(path)

if __name__ == '__main__':
    # ideally, would run as 'python predict.py model_type'
    model_type = sys.argv[1]
    seq_len = 30
    thresh = 0.5

    # read clustered data
    # 0=mid, 1=low, 2=high energy
    music = pd.read_csv('tracks_clustered_k3.csv', usecols=['cluster', 'energy', 'artists', 'valence', 'name'])

    low_energy = music[music['cluster']==1].reset_index()
    high_energy = music[music['cluster']==2].reset_index()

    # low_energy.hist('valence', bins=20)
    # plt.savefig('low_valence_hist.png')
    # plt.clf()

    # high_energy.hist('valence', bins=20)
    # plt.savefig('high_energy_hist.png')
    # plt.clf()


    # prepare testing data
    prefix = 'models/'
    df = dataset.prepare_dataset(extract_test=True).reset_index(drop=True)
    print(df)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['tweets'])
    data = pad_sequences(tokenizer.texts_to_sequences(df['tweets']), maxlen=seq_len)
    if model_type == "CNN":
        model = load_model(prefix + 'CNN.h5')
    else:
        model = load_model(prefix + 'LSTM.h5')

    probs = model.predict(data)
    probs[probs>thresh] = 1
    probs[probs<=thresh] = 0
    for ind, prob in enumerate(probs):
        if prob == 1:
            song = high_energy.sample(1)
            print('Detected happy sentiment: ')
            print(df['tweets'].iloc[ind])
            print('Suggested song: ')
            print(song['name'].iloc[0], ' ', song['artists'].iloc[0], ' with energy val=', song['energy'].iloc[0])
            print(' ------------- ')

        else:
            song = low_energy.sample(1)
            print('Detected sad sentiment: ')
            print(df['tweets'].iloc[ind])
            print('Suggested song: ')
            print(song['name'].iloc[0], ' ', song['artists'].iloc[0], ' with energy val=', song['energy'].iloc[0])
            print(' ------------- ')

        # print(prob, ' ', df['y'].iloc[ind], ' ', song['name'].iloc[0], ' ', song['artists'].iloc[0])

