import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers



def mnist_digit_data():
   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
   train_images  = train_images / 255.0
   test_images = test_images / 255.0
   return train_images, test_images, train_labels, test_labels



def mnist_model():
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28)))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))
    return model


def model_compile(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

def model_fit(model,  epochs,train_images,train_labels):
    model.fit(train_images, train_labels, epochs=epochs,verbose=1)
    return model


def model_evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels)
    return test_loss, test_acc