import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

pre_trained_model = tf.keras.applications.MobileNetV2(input_shape=(150,150,3),include_top=False,
                                                                   weights='imagenet')
# pre_trained_model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
pre_trained_model.trainable=False
print(len(pre_trained_model.non_trainable_weights))
# for layer in pre_trained_model.layers:
#     layer.trainable=False

# pre_trained_model.summary()

# last_layer = pre_trained_model.get_layer('mixed7')
# last_output = last_layer.output
# print(last_layer)
# print(last_output)

# x = tf.keras.layers.Flatten()(last_output)
# x = tf.keras.layers.Dense(1024,activation='relu')(x)
# x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(1,activation='sigmoid')(x)

model = tf.keras.models.Sequential([pre_trained_model,
                                    # tf.keras.layers.Flatten(),
                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.BatchNormalization(),
                                    # tf.keras.layers.Dense(512,activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(1,activation='sigmoid')])

print(pre_trained_model.input)
pre_trained_model.summary()
# model = tf.keras.Model(pre_trained_model.input,x)

model.summary()

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Define training and validation directory
train_dir = '../Week4_1/data_set_1'
val_dir = '../Week4_1/validation_data_set_1'

train_datagen = IDG(rescale=1/255.0,width_shift_range=0.2,height_shift_range=0.2,
                    zoom_range=0.2,horizontal_flip=True)
val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir,target_size=(150,150),batch_size=20,class_mode='binary')

# callback = tf.keras.callbacks.TensorBoard(log_dir='log_horse_human')


model.fit(train_generator,epochs=10,validation_data=val_generator)

model.save('transfer_learning_horse_human_mobilenetv2.h5')




