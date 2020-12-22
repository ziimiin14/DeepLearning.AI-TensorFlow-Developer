import numpy as np
import tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print('\nReached 99% accuracy so cancelling training!')
            self.model.stop_training=True


def train_mnist():
    # Callback function
    callbacks = myCallback()

    # Load data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize img to 0 to 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0


    # Create neural network
    model = tf.keras.Sequential([tf.keras.layers.Flatten(), # flatten into 1d
                                 tf.keras.layers.Dense(512,activation=tf.nn.relu), #hidden layer
                                 tf.keras.layers.Dense(10,activation=tf.nn.softmax)]) # output later (10 outputs)

    # Compile the model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model
    model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])

    model.evaluate(x_test,y_test)


train_mnist()



