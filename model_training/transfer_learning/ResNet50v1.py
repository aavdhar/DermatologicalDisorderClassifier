from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
import pickle

X = pickle.load(open('X2.pickle', 'rb'))
y = pickle.load(open('y2.pickle', 'rb'))

print('loaded dataset')

X = X/255.0


height = 150
width = 150


Name = "ResNet50-ImageNetWeightsNonTrainable"
tensorboard = TensorBoard(log_dir='logs\\test\\{}'.format(Name))


base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(height, width, 3))

for layer in base_model.layers:
    layer.trainable = False

flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(1024, activation='relu',)(flat1)
# output layer with 9 nodes for 9 classes
output = Dense(9, activation='softmax')(class1)

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


final_model.save('ResNet50v1.model')
