import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

# Reshape from (length,rol,col) to (length,row,col,channel)
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Normalize img data from 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# tf.keras.models.Sequential (another way)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
                             tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=128,activation='relu'),
                             tf.keras.layers.Dense(units=10,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)

model.evaluate(x_test,y_test)

