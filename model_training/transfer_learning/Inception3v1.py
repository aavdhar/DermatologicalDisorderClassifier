from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
import pickle


X = pickle.load(open('X2.pickle', 'rb'))
y = pickle.load(open('y2.pickle', 'rb'))

print('loaded dataset')

X = X/255.0


height = 150
width = 150


Name = "Inception3k-ImageNet_Weights_NonTrainable+Dropout"
tensorboard = TensorBoard(log_dir='logs\\test\\{}'.format(Name))


base_model = InceptionV3(weights='imagenet',
                      include_top=False,
                      input_shape=(height, width, 3))

for layer in base_model.layers:
    layer.trainable = False

flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu',)(flat1)
dropout = Dropout(.5)(class1)
class2 = Dense(1024, activation='relu',)(dropout)
dropout2 = Dropout(.5)(class2)
output = Dense(9, activation='softmax')(dropout2)

final_model = Model(inputs=base_model.inputs, outputs=output)
final_model.summary()


final_model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

final_model.fit(X, y,
          batch_size=32,
          epochs=10,
          validation_split=0.1,
          callbacks=[tensorboard])


final_model.save('ResNet50v2.model')
