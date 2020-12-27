import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
import matplotlib.pyplot as plt
import os



# Define model
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu',),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D((2,2)),
                                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D((2, 2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=512,activation='relu'),
                                    tf.keras.layers.Dense(units=1,activation='sigmoid')
                                    ])

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])

# Define train and validation generator
train_dir = '../Week1_2/dog_cat_train_test/training'
val_dir = '../Week1_2/dog_cat_train_test/testing'

train_datagen = IDG(rescale=1/255.0,rotation_range=72,width_shift_range=0.2,height_shift_range=0.2,
                    shear_range=0.1,zoom_range=0.3,horizontal_flip=True,fill_mode='nearest')
val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')

model.summary()

# model.fit_generator(train_generator,validation_data=val_datagen,steps_per_epoch=100,epochs=5,validation_steps=50)
model.fit(train_generator,validation_data=val_generator,epochs=15,steps_per_epoch=1125,validation_steps=125)
model.save('dog_cat_full_avoid_overfit.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()