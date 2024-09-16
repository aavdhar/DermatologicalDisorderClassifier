import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle

dataDir = "C:/Users/Aadi Dhar/Downloads/SkinDataset/Train"

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
def create_training_data():
    for cat in categories:
        path = os.path.join(dataDir, cat)
        class_num = categories.index(cat)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (imgSize, imgSize))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
print('length of training data:: \n', len(training_data))

random.shuffle(training_data)

X = []
y =[]
for features, label in training_data:
     X.append(features)
     y.append(label)

X = np.array(X).reshape(len(training_data), imgSize, imgSize, 3)
y = np.array(y)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

print('successfully dumped pickle files')