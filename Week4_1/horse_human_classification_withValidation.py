import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(300,300,3)),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    # Second convolution
                                    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    # Third convolution
                                    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    # Forth convolution
                                    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                                    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                    # Dense layers
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(units=512,activation='relu'),
                                    tf.keras.layers.Dense(units=1,activation='sigmoid')])

model.summary()

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])


train_datagen = IDG(rescale=1/255)
validation_datagen = IDG(rescale=1/255)

train_generator = train_datagen.flow_from_directory('data_set_1',
                                                    target_size=(300,300), # ALl images will be resized to 300x300
                                                    batch_size=128,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory('validation_data_set_1',
                                                              target_size=(300,300),
                                                              batch_size=32,
                                                              class_mode='binary')

history = model.fit(train_generator,steps_per_epoch=8,epochs=15,verbose=1,
                    validation_data=validation_generator,validation_steps=8)

model.save('horse_human_model_withValidation.h5')
