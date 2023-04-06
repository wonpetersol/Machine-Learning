import numpy as np
import pandas as pd
from dataset import *
from dec import *
from sentence_transformers import SentenceTransformer

def get_clustered_data():
    # set cluster number
    clusters=3
    # read original tracks data
    data, X, input_dim, sample_size = clean_data()
    # load previos model
    autoencoder, encoder = auto_encoder(input_dim=input_dim, layer_dim1=500, \
                                            layer_dim2=500, layer_dim3=2000, \
                                            latent_dim=10, 
                                            init=VarianceScaling(scale=1./3.,mode='fan_in',distribution='uniform'))
    autoencoder.load_weights('./models/autoencoder.h5')
    encoder.load_weights('./models/encoder.h5')
    clustering_layer = ClusteringLayer(clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    model.load_weights('./models/DEC.h5')
    # assign cluster
    pred, _ = model.predict(X, verbose=0)
    ypred = pred.argmax(axis = 1)
    data_clustered = data
    data_clustered['cluster'] = ypred
    
    return data_clustered
    
def prepare_dataset_tracks():
    # returns prepared dataset with cleaned music song title/name
    tracks_df = get_clustered_data()
    # check for null value
    tracks_df['name'] = tracks_df['name'].fillna(value=' ')
    tracks_df['original_name'] = tracks_df['name']
    # rename column 'name' as 'tweet' so we can reuse the same function and logic
    tracks_df.rename(columns={'name':'tweet'}, inplace=True)
    cleaned_tracks_df = clean(tracks_df)
    # tokenized, sw_removed, lemmatized are 2d lists
    tokenized_tracks = tokenize_tweets(cleaned_tracks_df)
    # sw_removed = remove_stopwords(tokenized)
    # lemmatized = lemmatize(sw_removed)
    lemmatized_tracks = lemmatize(tokenized_tracks)
    # merge into df for ease of use
    final_title = []
    for entry in lemmatized_tracks:
        final_title.append(' '.join(entry))
    d = {'id':tracks_df['id'],'cluster':tracks_df['cluster'],'name':tracks_df['original_name'],'cleaned_name':final_title,
         'valence':tracks_df['valence'],'energy':tracks_df['energy']}
    final_df = pd.DataFrame(data=d)
    return final_df

if __name__ == '__main__':
    # clean song title 
    tracks_df = prepare_dataset_tracks()
    # save data for future use
    #tracks_df.to_csv('tracks_title_cleaned.csv', index=False)
    
    # define sentence bert model
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # read data
    #tracks_df = pd.read_csv('./tracks_title_cleaned.csv')
    tracks_df['cleaned_name'] = tracks_df['cleaned_name'].fillna(' ')
    
    # low energy tracks
    low_tracks = tracks_df[tracks_df.cluster==1].reset_index(drop=True)
    # high energy tracks
    high_tracks = tracks_df[tracks_df.cluster==2].reset_index(drop=True)
    
    # graph the cluster distribution based on energy and valence
    low_tracks.hist('valence', bins=20, )
    low_tracks.hist('energy', bins=20)
    high_tracks.hist('valence', bins=20)
    high_tracks.hist('energy', bins=20)
    
    # subset cluster further
    # low cluster: energy < 0.6, valence < 0.6
    # high cluster: energy > 0.6, valence > 0.6
    low_tracks = tracks_df[((tracks_df.cluster==1) & (tracks_df.energy < 0.6) & (tracks_df.valence < 0.6))].reset_index(drop=True)
    high_tracks = tracks_df[((tracks_df.cluster==2) & (tracks_df.energy > 0.6) & (tracks_df.valence > 0.6))].reset_index(drop=True)
    
    # vectorized and save clean song title using sentenced bert
    low_vec = sbert_model.encode(low_tracks['cleaned_name'].values)
    np.savetxt('low_vec.csv',low_vec, delimiter=',')
    high_vec = sbert_model.encode(high_tracks['cleaned_name'].values)
    np.savetxt('high_vec.csv',high_vec, delimiter=',')
