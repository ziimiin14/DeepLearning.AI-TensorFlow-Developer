import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

pre_trained_model = tf.keras.applications.MobileNetV3Small(input_shape=(150,150,3),include_top=False,
                                                        weights='imagenet')
# pre_trained_model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
pre_trained_model.trainable=False
pre_trained_model.summary()
print(len(pre_trained_model.non_trainable_weights))

model = tf.keras.models.Sequential([pre_trained_model,
                                    # tf.keras.layers.Flatten(),
                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.BatchNormalization(),
                                    # tf.keras.layers.Dense(512,activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1,activation='sigmoid')])

# print(pre_trained_model.input)
# model = tf.keras.Model(pre_trained_model.input,x)

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['accuracy'])

# Define training and validation directory
train_dir = '../Week1_2/dog_cat_train_test/training'
val_dir = '../Week1_2/dog_cat_train_test/testing'

train_datagen = IDG(rescale=1/255.0,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,
                    shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')

# callback = tf.keras.callbacks.TensorBoard(log_dir='log')


model.fit(train_generator,epochs=3,validation_data=val_generator)




# model.save('dog_cat_partialds_inception.h5')



