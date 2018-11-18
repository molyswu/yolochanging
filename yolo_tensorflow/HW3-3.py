'''
 change the activation functions, the energy function, early stopping, and L2_2​2​​-regularization

'''

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping, TensorBoard

batch_size = 8192
epoches = 200
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

save_dir='output'
model_name = 'keras_dense_mnist'

model = Sequential()
model.add(Dense(units=100,activation='relu',input_dim=784))
model.add(Dense(units=100,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt, metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False),

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train, epochs = epoches, batch_size = batch_size,validation_data=(x_test, y_test), shuffle=True,callbacks = earlystop)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])