import pickle
import numpy as np
import math
import tensorflow as t
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import model_from_json
import json
import os

pickleFile = open('behave.pickle', 'rb')
data = pickle.load(pickleFile)
X_features = data['features']
Y_labels = data['labels']
from sklearn.utils import shuffle
features, labels = shuffle(X_features, Y_labels)

model_json = "model.json"
model_weights = "model.h5"
height = 32
width = 64
model = Sequential()
# Kernel Size 3 x 3 (16 Filters)

model.add(Convolution2D(16, 3, 3, input_shape=(height, width, 3), border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

# Kernel Size 3 x 3 (32 Filters)
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten)

model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.compile('adam', 'mean_squared_error', ['accuracy'])

print(model.summary())

history = model.fit(features, labels, nb_epoch=20, validation_split=0.3)

model_as_json = model.to_json()
try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass
with open(model_json, 'w') as output:
    json.dump(model_as_json, output)
model.save_weights(model_weights)


