import os
from random import randint

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

count_list = []
for cat in categories:
    path = os.path.join(dataDir, cat)
    count = 0
    for img in os.listdir(path):
        count += 1
    print(cat+':', count)
    count_list.append(count)

small = min(count_list)
print('\nsmallest number of images in a certain class::',small)

for cat in categories:
    while True:
        path = os.path.join(dataDir, cat)
        imgList = os.listdir(path)
        os.remove(os.path.join(path, imgList[0]))
        if len(imgList) == small:
            break


for cat in categories:
    path = os.path.join(dataDir, cat)
    count = 0
    for img in os.listdir(path):
        count += 1
    print(cat+':', count)