import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from keras.utils import plot_model
from IPython.display import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.random.set_seed(10)
#%%
# define autoencoder model
def auto_encoder(input_dim, layer_dim1, layer_dim2, layer_dim3, latent_dim, init, activation='relu'):
    
    input_data = Input(shape=(input_dim,), name='input')
    x = input_data
    
    # encoder  layer
    x = Dense(layer_dim1, activation=activation, kernel_initializer=init, name="encoder1")(x)
    x = Dense(layer_dim2, activation=activation, kernel_initializer=init, name="encoder2")(x)
    x = Dense(layer_dim3, activation=activation, kernel_initializer=init, name="encoder3")(x)

    # latent hidden layer (named as part of encoder layer)
    x = Dense(latent_dim, kernel_initializer=init, name="encoder4")(x)

    # encoder model
    encoder = Model(inputs=input_data, outputs=x, name='encoder')
    
    # decoder layer
    x = Dense(layer_dim3, activation=activation, kernel_initializer=init, name="decoder4")(x)
    x = Dense(layer_dim2, activation=activation, kernel_initializer=init, name="decoder3")(x)
    x = Dense(layer_dim1, activation=activation, kernel_initializer=init, name="decoder2")(x)

    # decoder output
    x = Dense(input_dim, kernel_initializer=init, name='decoder1')(x)
    
    # autoencoder model
    autoencoder = Model(inputs=input_data, outputs=x, name='autoencoder')
    
    return autoencoder, encoder

# Define a clustering layer for DEC
class ClusteringLayer(Layer):
    '''
    Clustering layer to converts input samples to soft label based on its feauters, by calculating
    the probability that the sample belong to a cluster based on student's t-distirbution used
    in t-SNE algorithm.
    '''

    def __init__(self, n_clusters, name, weights=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs):
        ''' 
        student t-distribution, as used in t-SNE algorithm.
        It measures the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
       
        inputs: the variable containing data, shape=(n_samples, n_features)
        
        Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        '''
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# define auxiliary target distribution for joint model training 
def target_distribution(pred):
    weight = pred ** 2 / pred.sum(axis = 0)
    return (weight.T / weight.sum(axis = 1)).T

def clean_data():
    # read data
    data = pd.read_csv('./data/tracks.csv')
    # filtered data to select columns to be used for clustering
    data_filtered = data.drop(['id','name','popularity','duration_ms','artists','id_artists','release_date'], axis=1)
    col_filtered = data_filtered.columns.values.tolist()
    scaler = MinMaxScaler() 
    # rescale all columns to have the same scale
    data_filtered[col_filtered] = scaler.fit_transform(data_filtered[col_filtered])
    # reshape to array
    X = data_filtered.values
    # number of columns used
    input_dim = X.shape[-1]
    sample_size = X.shape[0]
    
    return data, X, input_dim, sample_size

def kmeans_elbow(X):
    # Determine initial number of cluster for soft labelling
    ssd = []
    for num_clusters in range(2,10):
        clusterer = KMeans(n_clusters=num_clusters)
        clusterer = clusterer.fit(X)
        #centers = clusterer.cluster_centers_
        ssd.append(clusterer.inertia_)
        print ("For num_clusters = {}, K-means sum of squares distance is {}".format(num_clusters, clusterer.inertia_))
    # plot elbow
    plt.plot(range(2,10), ssd, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum of squares distance')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def initialize_autoencoder(X, input_dim, kernel_init, epochs, batch_size, optimizer, save_dir):
    # initialize autoencoder and encoder with 3 encoder/decoder layer and 1 hidden latent layer
    autoencoder, encoder = auto_encoder(input_dim=input_dim, layer_dim1=500, \
                                        layer_dim2=500, layer_dim3=2000, \
                                        latent_dim=10, init=kernel_init)
    
    # plot model
    #plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)
    #Image(filename='autoencoder.png') 
    #plot_model(encoder, to_file='encoder.png', show_shapes=True)
    #Image(filename='encoder.png') 
    
    # Pre-train model to preprocessed data, with mse as loss function
    autoencoder.compile(optimizer=optimizer, loss='mse')
    autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs)
    autoencoder.save_weights(save_dir + '/autoencoder.h5')
    encoder.save_weights(save_dir + '/encoder.h5')
    
    return autoencoder, encoder

def run_kmeans_cluster(clusters,X,encoder):
    # run cluster using k-means to get initial cluster centroids using encoded data
    kmeans = KMeans(n_clusters=clusters, n_init=20)
    ypred = kmeans.fit_predict(encoder.predict(X))
    
    return kmeans, ypred

def initialize_dec(clusters, autoencoder, encoder, kmeans, optimizer):
    # add clustering layer to autoencoder model
    clustering_layer = ClusteringLayer(clusters, name='clustering')(encoder.output)
    # Define a joint model that takes the preprocessed tracks dataset as input and 
    # output predicted clusters and decoded input data 
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    model.summary()
    
    # plot new joint model
    #plot_model(model, to_file='model.png', show_shapes=True)
    #Image(filename='model.png')
    
    # set clustering layer weights based on inital cluster centroids from k-means
    clustering_layer_name = [layer.name for layer in model.layers if "clustering" in layer.name][0]
    model.get_layer(name=clustering_layer_name).set_weights([kmeans.cluster_centers_])
    
    # re-compile model with KL divergence loss and mse as loss function
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)

    return model

def train_dec(model, optimizer, ypred, sample_size, X, batch_size, save_dir):
    # re-compile model with KL divergence loss and mse as loss function
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=optimizer)
    
    # set tolerance, max iteration, and update interval
    tol = 0.001
    loss = 0
    index = 0
    max_iter = 1000
    update_interval = 100
    index_array = np.arange(sample_size)
    ypred_prev = ypred
    # train DEC iteratively by
    # iteratively refines the clusters by learning from the high confidence assignments 
    # with the help of the auxiliary target distribution
    # by matching the soft assignment to the target distribution
    # based on KL divergence loss between the soft label and the auxiliary target distribution
    for iter_num in range(int(max_iter)):
        if iter_num % update_interval == 0:
            print("Iteration: %s" % iter_num)
            # get vector of probability across all clusters
            pred, _  = model.predict(X, verbose=0)
            # update target distribution
            prob = target_distribution(pred)
            # set cluster assignment based on highest probability number
            ypred = pred.argmax(1)
            # check stopping criteria
            # if the label difference is less than the tolerance between previous and current iteration
            # stop training
            label_diff = np.sum(ypred != ypred_prev).astype(np.float32) / ypred.shape[0]
            ypred_prev = ypred
            if iter_num > 0 and label_diff < tol:
                print('label difference', label_diff, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, sample_size)]
        loss = model.train_on_batch(x=X[idx], y=[prob[idx], X[idx]])
        index = index + 1 if (index + 1) * batch_size <= sample_size else 0
    
    # save model for future reference
    model.save_weights(save_dir + '/DEC.h5')
    
    return model

def main():
    print("Getting data")
    # get clean data
    data, X, input_dim, sample_size = clean_data()
    
    print("Plot K-means elbow to determine inital clusters")
    # plot kmeans elbow to determine number of clusters
    kmeans_elbow(X)
    
    # set cluster number
    clusters = 3
    # set number of epoch, batch size, and initialization for model and training
    epochs = 20
    batch_size = 64
    kernel_init = VarianceScaling(scale=1./3.,mode='fan_in',distribution='uniform')
    # using SGD optimizer
    optimizer = SGD(lr=1, momentum=0.9)
    # directory to save model results
    save_dir = './results'

    print("Initialize Autoencoder")
    # initialize autoencoder, encoder, and kmeans cluster center for clustering layer
    autoencoder, encoder = initialize_autoencoder(X, input_dim, kernel_init, epochs, batch_size, optimizer,save_dir)
    kmeans, ypred = run_kmeans_cluster(clusters,X,encoder)
    
    print("Initialize DEC")
    # initialize DEC model by incorporating clustering layer
    model = initialize_dec(clusters, autoencoder, encoder, kmeans, optimizer)
    print("Training DEC")
    # train DEC model iteratively
    model = train_dec(model, optimizer, ypred, sample_size, X, batch_size, save_dir)
    
    # use the model to assign predicted cluster and update target distribution
    pred, _ = model.predict(X, verbose=0)
    prob = target_distribution(pred)  
    ypred = pred.argmax(axis = 1)
    
    # add cluster label to dataset
    data_clustered = data
    data_clustered['cluster'] = ypred
        
    # save data for future use
    data_clustered.to_csv('tracks_clustered.csv', index=False)
    
    # print distribution of clusters
    for i in range(clusters):
        print("Cluster %s" % i)
        clusters_temp = data_clustered[data_clustered['cluster'] == i]
        print(clusters_temp[['energy','acousticness','valence','tempo']].describe(percentiles=[.5]))

#%%
if __name__ == '__main__':
    main()
