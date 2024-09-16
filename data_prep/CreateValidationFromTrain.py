import os
import shutil

dataDir = "C:/Users/Aadi Dhar/Downloads/SkinDataset/Train2Val"
targetDir = 'C:/Users/Aadi Dhar/Downloads/SkinDataset/Validation2'

categories = ['Acne or Rosacea',
              'Acnitic Keratosis',
              'Eczema',
              'Melanoma',
              'Poison Ivy',
              'Psoriasis',
              'Seborrheic Keratoses',
              'Tinea Ringworm',
              'Warts, Molluscum, Other Viral Infections']

# Make categories
'''
for item in categories:
    os.mkdir(os.path.join(targetDir, item))
'''

for cat in categories:
    path = os.path.join(dataDir, cat)
    print(cat)
    count = 0
    for img in os.listdir(path):
        if count == 99:
            break
        shutil.move(os.path.join(path, img), os.path.join(targetDir, cat))
        count += 1
