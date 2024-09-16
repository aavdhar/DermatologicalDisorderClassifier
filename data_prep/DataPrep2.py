import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


print('program is running')

dataDir = "C:/Users/Aadi Dhar/Downloads/SkinDataset/Train2"


categories = ['Acne or Rosacea',
              'Acnitic Keratosis',
              'Eczema',
              'Melanoma',
              'Poison Ivy',
              'Psoriasis',
              'Seborrheic Keratoses',
              'Tinea Ringworm',
              'Warts, Molluscum, Other Viral Infections']

imgSize = 150

training_data = []
def vertical(filepath, classification, reps):
    img = load_img(filepath)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(height_shift_range=[-200, 200])
    it = datagen.flow(samples, batch_size=1)
    for i in range(reps):
        plt.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        data_image = img_to_array(image)
        data_resized = cv2.resize(data_image, (imgSize, imgSize))
        training_data.append([data_resized, classification])

def rotate(filepath, classification, reps):
    img = load_img(filepath)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(width_shift_range=[-200, 200])
    it = datagen.flow(samples, batch_size=1)
    for i in range(reps):
        plt.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        data_image = img_to_array(image)
        data_resized = cv2.resize(data_image, (imgSize, imgSize))
        training_data.append([data_resized, classification])

def create_training_data():
    for cat in categories:
        path = os.path.join(dataDir, cat)
        class_num = categories.index(cat)
        print(cat)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (imgSize, imgSize))
            training_data.append([new_array, class_num])

            vertical(os.path.join(path, img), class_num, 4)

            rotate(os.path.join(path, img), class_num, 4)


create_training_data()
print('len of training data is:\n', len(training_data))


random.shuffle(training_data)


def convert_export(td):
    X = []
    y = []

    for features, label in td:
         X.append(features)
         y.append(label)

    X_aug = np.array(X).reshape(-1, imgSize, imgSize, 3)
    y_aug = np.array(y)

    pickle_out = open('X2.pickle', 'wb')
    pickle.dump(X_aug, pickle_out, protocol=4)
    pickle_out.close()

    pickle_out = open('y2.pickle', 'wb')
    pickle.dump(y_aug, pickle_out, protocol=4)
    pickle_out.close()

    print('successfully dumped pickle files')


convert_export(training_data)