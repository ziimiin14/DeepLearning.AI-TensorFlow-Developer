import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import os

# Define directory
train_dir = 'dog_cat_partial_dataset/cats_and_dogs_filtered/train'
val_dir = 'dog_cat_partial_dataset/cats_and_dogs_filtered/validation'

# Define model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu',),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=512,activation='relu'),
                                    tf.keras.layers.Dense(units=1,activation='sigmoid')
                                    ])

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])

# Define train and validation generator
train_datagen = IDG(rescale=1/255.0)
val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')

model.summary()

# model.fit_generator(train_generator,validation_data=val_datagen,steps_per_epoch=100,epochs=5,validation_steps=50)
model.fit(train_generator,validation_data=val_generator,steps_per_epoch=100,epochs=5,validation_steps=50)
model.save('dog_cat_partial.h5')
