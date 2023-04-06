import pandas as pd
import dataset
import matplotlib.pyplot as plt
import tensorflow.keras.models as tfk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import sys

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_tracks import *

def load_model(path):
    return tfk.load_model(path)

if __name__ == '__main__':
    # ideally, would run as 'python predict.py model_type'
    model_type = sys.argv[1]
    seq_len = 30
    thresh = 0.5

    # read clustered data
    # 0=mid, 1=low, 2=high energy
    #music = pd.read_csv('tracks_clustered_k3.csv', usecols=['cluster', 'energy', 'artists', 'valence', 'name'])

    #low_energy = music[music['cluster']==1].reset_index()
    #high_energy = music[music['cluster']==2].reset_index()
    
    # low_energy.hist('valence', bins=20)
    # plt.savefig('low_valence_hist.png')
    # plt.clf()

    # high_energy.hist('valence', bins=20)
    # plt.savefig('high_energy_hist.png')
    # plt.clf()
    
    # define sentence bert model
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # read clustered tracks data and vectors 
    # (the vectors data is obtained by running dataset_tracks.py)
    tracks_df = get_clustered_data()
    # low energy tracks
    low_tracks = tracks_df[((tracks_df.cluster==1) & (tracks_df.energy < 0.6) & (tracks_df.valence < 0.6))].reset_index(drop=True)
    low_vec = pd.read_csv('./low_vec.csv',header=None).values
    # high energy tracks
    high_tracks = tracks_df[((tracks_df.cluster==2) & (tracks_df.energy > 0.6) & (tracks_df.valence > 0.6))].reset_index(drop=True)
    high_vec = pd.read_csv('./high_vec.csv',header=None).values
    
    n = 5 # number of song recommendations

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
        model = load_model(prefix + 'RNN.h5')

    probs = model.predict(data)
    probs[probs>thresh] = 1
    probs[probs<=thresh] = 0
    for ind, prob in enumerate(probs):
        if prob == 1:
            #song = high_energy.sample(1)
            # vectorize sample tweets using sentenced bert
            input_tweet_vec = sbert_model.encode([df['tweets'].iloc[ind]])
            # calculate cosine similarity between song titles and input tweet
            # suggest top 5 songs with highest score
            cos_sim = cosine_similarity(input_tweet_vec, high_vec)
            song_rec = high_tracks['name'].values[(-cos_sim).argsort()[0][:n]]
            artists_rec = high_tracks['artists'].values[(-cos_sim).argsort()[0][:n]]
            print('Detected happy sentiment: ')
            print(df['tweets'].iloc[ind])
            print('Suggested song: ')
            for idx in range(n):
                print("%s by %s" % (song_rec[idx],artists_rec[idx]))
            #print(song['name'].iloc[0], ' ', song['artists'].iloc[0], ' with energy val=', song['energy'].iloc[0])
            print(' ------------- ')

        else:
            #song = low_energy.sample(1)
            # vectorize sample tweets using sentenced bert
            input_tweet_vec = sbert_model.encode([df['tweets'].iloc[ind]])
            # calculate cosine similarity between song titles and input tweet
            # suggest top 5 songs with highest score
            cos_sim = cosine_similarity(input_tweet_vec, low_vec)
            song_rec = low_tracks['name'].values[(-cos_sim).argsort()[0][:n]]
            artists_rec = low_tracks['artists'].values[(-cos_sim).argsort()[0][:n]]
            print('Detected sad sentiment: ')
            print(df['tweets'].iloc[ind])
            print('Suggested song: ')
            for idx in range(n):
                print("%s by %s" % (song_rec[idx],artists_rec[idx]))
            #print(song['name'].iloc[0], ' ', song['artists'].iloc[0], ' with energy val=', song['energy'].iloc[0])
            print(' ------------- ')

        # print(prob, ' ', df['y'].iloc[ind], ' ', song['name'].iloc[0], ' ', song['artists'].iloc[0])

