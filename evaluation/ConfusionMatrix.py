import cv2
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

dataDir = "C:/Users/Aadi Dhar/Downloads/SkinDataset/test2"
imgSize = 150
CATEGORIES = ['Acne or Rosacea',
              'Acnitic Keratosis',
              'Eczema',
              'Melanoma',
              'Poison Ivy',
              'Psoriasis',
              'Seborrheic Keratoses',
              'Tinea Ringworm',
              'Warts, Molluscum, Other Viral Infections']

CategoriesFormatted = [
    'Acne\nor Rosacea',
  'Acnitic\nKeratosis',
  'Eczema',
  'Melanoma',
  'Poison Ivy',
  'Psoriasis',
  'Seborrheic Keratoses',
  'Tinea\nRingworm',
  'Warts,\nMolluscum,\nOther\nViral Infections']


model = load_model('5-128-5-CNN2.model')


score = 0
count = 0
guess_list = []
answer_list = []
for category in CATEGORIES:
    path = os.path.join(dataDir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        new_array = cv2.resize(img_array, (imgSize, imgSize))
        image = new_array.reshape(-1, imgSize, imgSize, 3)
        prediction = model.predict([image])
        prediction = list(prediction[0])
        prediction = CATEGORIES[prediction.index(max(prediction))]
        print('guess::  ', prediction)
        print('answer::  ', category, '\n----------------------------')
        guess_list.append(prediction)
        answer_list.append(category)
        if str(prediction) == str(category):
            score += 1
        count += 1


print('score::', score)
print('count::', count)
print('raw result::', score/count)

cm = confusion_matrix(answer_list, guess_list, labels=CATEGORIES)
print(cm)

df_cm = pd.DataFrame(cm, CategoriesFormatted, CategoriesFormatted)

sn.set(font_scale=.95)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')

plt.show()
