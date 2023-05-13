import tensorflow as tf
from keras import layers,models
import numpy as np




def cifar100_data():
   ( train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
   train_images, test_images = train_images / 255.0, test_images / 255.0
   return train_images, train_labels,test_images, test_labels

def cifar100_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(100), activation='softmax')
    return model

def model_compile(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    return model


def model_fit(model,epochs, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=epochs)
    return model

def model_evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,test_labels)
    return test_loss,test_acc
