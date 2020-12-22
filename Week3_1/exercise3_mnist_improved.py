import numpy as np
import tensorflow as tf

#
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998):
            print('\n Reached above 99.8%')
            self.model.stop_training = True


# Callback function to stop after accuracy reached 99.8%
callbacks = myCallback()


# Load data
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Reshape from (length,rol,col) to (length,row,col,channel)
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Normalize img data from 0 to 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# tf.keras.models.Sequential (another way)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
                             tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=128,activation='relu'),
                             tf.keras.layers.Dense(units=10,activation='softmax')])



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()

model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])

model.evaluate(x_test,y_test)

