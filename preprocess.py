import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imresize
from six.moves import cPickle as pickle
import cv2

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


def read_image(line, root):
    extracted = extract_features(line)
    img_path = extracted[0]
    path = root + "" + img_path
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
    return x, -round(float(y),3)

def save_file(root_data, pickle_file, limit=500):
    log_file = root_data + "driving_log.csv"
    with open(log_file, 'r') as reader:
        lines = reader.readlines()
        total_size = len(lines)-2
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        print("Starting to extract data")
        print("Extracting a total of ", 2*total_size)
        zero_counter = 0
        for i in range(1, total_size):
            try:
                x, y = read_image(lines[i], root_data)
                y = round(float(y), 3)
                if y == 0:
                    zero_counter += 1
                    if zero_counter > limit:
                        continue
                x = pipeline(x)

                X1.append(x)
                Y1.append(y)

                x, y = flip_image(x, y)
                X2.append(x)
                Y2.append(y)
            except Exception as e:
                print(e)
        print("Created images")

    print("zeros counter", zero_counter)
    print("Extracted images:", len(X1))

    features = np.append(np.array(X1), np.array(X2), axis=0)
    labels = np.append(np.array(Y1), np.array(Y2), axis=0)

    data = {
        'features': features,
        'labels': labels
    }

    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print("Data normal successfully saved")

    except Exception as e:
        print("Unable to save: ",e)
