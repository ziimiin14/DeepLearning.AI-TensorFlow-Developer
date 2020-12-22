import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(300,300,3)),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=512,activation='relu'),
                                    tf.keras.layers.Dense(units=1,activation='sigmoid')])
model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'],loss='binary_crossentropy')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory('data_set_1',target_size=(300,300),batch_size=128,class_mode='binary')

history = model.fit(train_generator, steps_per_epoch=8,epochs=15)

model.save('horse_human_model')