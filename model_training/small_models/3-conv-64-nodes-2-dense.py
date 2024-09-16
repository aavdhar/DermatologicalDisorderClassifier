from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle

X = pickle.load(open('X_aug_v2.pickle', 'rb'))
y = pickle.load(open('y_aug_v2.pickle', 'rb'))

print('loaded dataset')

X = X/255.0


Name = "3-64-2aug"
tensorboard = TensorBoard(log_dir='logs\\test\\{}'.format(Name))

model = Sequential()
# 3 convolutional layers
model.add(Conv2D(64, (3, 3), input_shape=(150,150,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Dense Layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

# The output layer with 18 neurons, for 18 classes
model.add(Dense(18))
model.add(Activation("softmax"))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=32,
          epochs=10,
          validation_split=0.1,
          callbacks=[tensorboard])


model.save('3-64-2-CNN-aug.model')