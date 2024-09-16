import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle



X = pickle.load(open('X_aug.pickle', 'rb'))
y = pickle.load(open('y_aug.pickle', 'rb'))

X = X/255.0

dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
for dense in dense_layers:
    for layer in layer_sizes:
        for conv in conv_layers:
            Name = "{}-conv-{}-nodes-{}-dense".format(conv, layer, dense)
            tensorboard = TensorBoard(log_dir='logs\\fit\\{}'.format(Name))

            model = Sequential()
            # 3 convolutional layers
            model.add(Conv2D(layer, (3, 3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv-1):
                model.add(Conv2D(layer, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

                model.add(Conv2D(layer, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Dropout(0.25))

            model.add(Flatten())
            for i in range(dense):
                model.add(Dense(layer))
                model.add(Activation('relu'))

            model.add(Dense(128))
            model.add(Activation("relu"))

            model.add(Dense(128))
            model.add(Activation("relu"))

            # The output layer with 18 neurons, for 18 classes
            model.add(Dense(18))
            model.add(Activation("softmax"))

            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            model.fit(X, y,
                      batch_size=32,
                      epochs=15,
                      validation_split=0.3,
                      callbacks=[tensorboard])
