import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


print('program is running')

dataDir = "C:/Users/Aadi Dhar/Downloads/SkinDataset/TrainEqual"

categories = ['Acne or Rosacea',
              'Acnitic Keratosis',
              'Atopic Dermatitis',
              'Bullous Disease',
              'Cellulitis Impetigo',
              'Eczema',
              'Lupus (or other Connective Tissue disorders)',
              'Melanoma',
              'Poison Ivy',
              'Psoriasis',
              'Scabies, Lyme Disease, Other Insect-Related',
              'Seborrheic Keratoses',
              'Systemic Disease',
              'Tinea Ringworm',
              'Urticaria Hives',
              'Vascular Tumors',
              'Vasculitis',
              'Warts, Molluscum, Other Viral Infections']

imgSize = 150

training_data = []
def horizontally_shift(filepath, classification, reps):
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


def zoom(filepath, classification, reps):
    img = load_img(filepath)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(zoom_range=[.5, 1.0])
    it = datagen.flow(samples, batch_size=1)
    for i in range(reps):
        plt.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        data_image = img_to_array(image)
        data_resized = cv2.resize(data_image, (imgSize, imgSize))
        training_data.append([data_resized, classification])


def bright(filepath, classification, reps):
    img = load_img(filepath)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(brightness_range=[.2, 1.0])
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

            horizontally_shift(os.path.join(path, img), class_num, 8)

            rotate(os.path.join(path, img), class_num, 8)

            zoom(os.path.join(path, img), class_num, 8)

            bright(os.path.join(path, img), class_num, 8)


create_training_data()
print('len of training data is:\n', len(training_data))


random.shuffle(training_data)


def convert_export(td):
    X_equal = []
    y_equal = []

    for features, label in td:
         X_equal.append(features)
         y_equal.append(label)

    X_equal = np.array(X_equal).reshape(-1, imgSize, imgSize, 3)
    y_equal = np.array(y_equal)

    pickle_out = open('X_aug_equal.pickle', 'wb')
    pickle.dump(X_equal, pickle_out, protocol=4)
    pickle_out.close()

    pickle_out = open('y_aug_equal.pickle', 'wb')
    pickle.dump(y_equal, pickle_out, protocol=4)
    pickle_out.close()

    print('successfully dumped pickle files')


convert_export(training_data)
