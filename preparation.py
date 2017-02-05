from six.moves import cPickle as pickle
import numpy as np

import preprocess
import model

origin_pickle = "origin.pickle"
delta_pickle = "delta.pickle"

def save():
    #preprocess.save_file("origin/data/", origin_pickle )
    preprocess.save_file("delta/", delta_pickle, limit=130 )


def train():
    pickle_1 = open(origin_pickle, 'rb')
    data_1 = pickle.load(pickle_1)
    pickle_2 = open(delta_pickle, 'rb')
    data_2 =pickle.load(pickle_2)

    X1 = data_1['features']
    Y1 = data_1['labels']
    X2 = data_2['features']
    Y2 = data_2['labels']

    X = np.append(X1, X2, axis=0)
    Y = np.append(Y1, Y2, axis=0)
    model.train(X, Y)

save()
train()