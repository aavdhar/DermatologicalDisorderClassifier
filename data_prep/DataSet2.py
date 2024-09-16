import os
from matplotlib import pyplot as plt
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import cv2
from random import randint


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



count_list = []
for cat in categories:
    path = os.path.join(dataDir, cat)
    count = 0
    for img in os.listdir(path):
        count += 1
    print(cat+':', count)
    count_list.append(count)

large = max(count_list)
small = min(count_list)
print('\nsmallest number of images in a certain class::', small)
print('\nlargest number of images in a certain class::', large)
print('\n\n_______________________________________________________________________\n\n')

def save_copy(img_path, save_directory, name):
    global datagen

    img = load_img(img_path)
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    types = ['rotation', 'horizontal', 'vertical', 'bright']
    rand = randint(0, 3)
    if types[rand] == 'rotation':
        datagen = ImageDataGenerator(rotation_range=90)
    elif types[rand] == 'horizontal':
        datagen = ImageDataGenerator(width_shift_range=[-200, 200])
    elif types[rand] == 'vertical':
        datagen = ImageDataGenerator(height_shift_range=0.5)
    elif types[rand] == 'bright':
        datagen = ImageDataGenerator(brightness_range=[.2, 1.0])
    it = datagen.flow(samples, batch_size=1)
    for i in range(1):
        plt.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        cv2.imwrite(save_directory+'/'+name,  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


for cat in categories:
    count = 0
    loop = 1
    print(cat)
    path = os.path.join(dataDir, cat)
    imgList = os.listdir(path)
    if len(imgList) < 1724:
        for img in imgList:
            if len(os.listdir(path)) == 1724:
                break
            if count == len(imgList) - 1:
                loop += 1
                print(loop)
            save_copy(os.path.join(path, img), path, cat+'x'+str(loop)+'x'+str(count)+'.jpg')
            count += 1
    else:
        continue




count_list2 = []
for cat in categories:
    path = os.path.join(dataDir, cat)
    count = 0
    for img in os.listdir(path):
        count += 1
    print(cat+':', count)
    count_list2.append(count)

large2 = max(count_list2)
small2 = min(count_list2)
print('\nsmallest number of images in a certain class::', small2)
print('\nlargest number of images in a certain class::', large2)
