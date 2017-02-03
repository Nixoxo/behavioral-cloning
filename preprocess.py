import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imresize
from six.moves import cPickle as pickle
import cv2

root_data = "data/"
log_file = root_data + "driving_log.csv"
pickle_file = "behave.pickle"
width = 64
height = 32


def extract_features(line):
    data = line.replace("\n", "").split(",")
    center = data[0]
    steering = data[3]
    throttle = data[4]
    brake = data[5]
    speed = data[6]
    return center, steering, throttle, brake, speed


def read_image(line):
    extracted = extract_features(line)
    img_path = extracted[0]
    path = root_data + "" + img_path
    data = plt.imread(path)
    label = extracted[1]
    return data, label

def normalize(x):
    return (x-127.5)/127.5

def pipeline(img):
    x = img[45:]
    x = normalize(x)
    x = imresize(x, (height, width))
    return x

def flip_image(x, y):
    x = cv2.flip(x, 1)
    return x, -float(y)

def save_file():
    with open(log_file, 'r') as reader:
        lines = reader.readlines()
        total_size = len(lines)-2
        X = np.zeros(shape=(2*total_size, height, width, 3))
        Y = np.zeros(shape=(2*total_size))
        print("Starting to extract data")
        print("Extracting a total of ", 2*total_size)
        for i in range(1, total_size):
            try:
                x, y = read_image(lines[i])
                x = pipeline(x)
                X[i-1] = x
                Y[i-1] = round(float(y),3)
            except Exception as e:
                print(e)
        print("created normal images")
        for i in range(1, total_size):
            try:
                x, y, read_image(lines[i])
                x, y = flip_image(x, y)
                X[total_size + i -1] = x
                Y[total_size + i -1] = round(float(y), 3)
            except Exception as e:
                print(e)
        print("created flipped images")


    print(len(X))
    print(len(Y))
    data = {
        'features': X,
        'labels': Y
    }

    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print("Data successfully saved")
    except Exception as e:
        print("Unable to save: ",e)

save_file()