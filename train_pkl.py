import warnings
import os

import pickle as pkl
import tensorflow as tf
from utils import plot_history
from keras.models import Sequential
from keras.utils import to_categorical
from keras import layers, optimizers, regularizers

warnings.simplefilter('ignore', FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_SIZE = 300
EPOCHS = 5
N_CLASSES = 6
VAL_BATCH_SIZE = 6
TRAIN_BATCH_SIZE = 18

with open('train_data.pkl', 'rb') as f:
    train_values = pkl.load(f)
    train_labels = pkl.load(f)

with open('validation_data.pkl', 'rb') as f:
    val_values = pkl.load(f)
    val_labels = pkl.load(f)

with open('test_data.pkl', 'rb') as f:
    test_values = pkl.load(f)
    test_labels = pkl.load(f)

train_labels = to_categorical(train_labels)
train_values = train_values.reshape(-1, 200, 200, 1)
train_values /= 255.0
print(f'Train data shape: {train_values.shape},{train_labels.shape}')

val_labels = to_categorical(val_labels)
val_values = val_values.reshape(-1, 200, 200, 1)
val_values /= 255.0
print(f'Validation data shape: {val_values.shape},{val_labels.shape}')

test_labels = to_categorical(test_labels)
test_values = test_values.reshape(-1, 200, 200, 1)
test_values /= 255.0
print(f'Test data shape: {test_values.shape},{test_labels.shape}')

model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu',
                        activity_regularizer=regularizers.l2(1e-5)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', activity_regularizer=regularizers.l2(1e-5)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', activity_regularizer=regularizers.l2(1e-5)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu', activity_regularizer=regularizers.l2(1e-5)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(N_CLASSES, activation='softmax'))

sgd = optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.0001)
rmsprop = optimizers.RMSprop(lr=0.0005)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

history = model.fit(x=train_values, y=train_labels, batch_size=TRAIN_BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    validation_data=(val_values, val_labels))

print(model.evaluate(test_values, test_labels))
plot_history(history)
model.save('model.h5')
