import numpy as np
import tensorflow as tf



fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img,train_labels),(test_img,test_labels) = fashion_mnist.load_data()

# Normalize the pixel value from 0 to 1 since nn works better in between 0 and 1
train_img = train_img / 255
test_img = test_img / 255

model = tf.keras.Sequential([tf.keras.layers.Flatten(), # input img is 28x28 pixel
                             tf.keras.layers.Dense(512,activation=tf.nn.relu), # hidden layer
                             tf.keras.layers.Dense(10,activation=tf.nn.softmax)]) # there are 10 labels outpu

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_img,train_labels,epochs=10)

model.evaluate(test_img,test_labels)