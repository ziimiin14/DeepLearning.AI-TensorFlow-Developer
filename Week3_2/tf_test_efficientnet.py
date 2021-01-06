import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

pre_trained_model = tf.keras.applications.EfficientNetB0(input_shape=(150,150,3),include_top=False,
                                                        weights='None')
pre_trained_model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
pre_trained_model.trainable=False
# pre_trained_model.summary()
# print(len(pre_trained_model.non_trainable_weights))

# inputs = tf.keras.Input(shape=(150,150,3))
# x = pre_trained_model(inputs,training=False)
# print('input = ', inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(pre_trained_model.output)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Flatten()(pre_trained_model.output)
x = tf.keras.layers.Dense(1024,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(pre_trained_model.input,outputs)


# print(pre_trained_model.input)
# model = tf.keras.Model(pre_trained_model.input,x)

model.summary()

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# Define training and validation directory
train_dir = '../Week1_2/dog_cat_partial_dataset/cats_and_dogs_filtered/train'
val_dir = '../Week1_2/dog_cat_partial_dataset/cats_and_dogs_filtered/validation'

train_datagen = IDG(rescale=1/255.0,rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,
                    shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')
# callback = tf.keras.callbacks.TensorBoard(log_dir='log')


model.fit(train_generator,epochs=20,validation_data=val_generator)




# model.save('dog_cat_partialds_inception.h5')



