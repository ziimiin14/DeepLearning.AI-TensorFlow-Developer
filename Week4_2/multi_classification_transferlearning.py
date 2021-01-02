import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

# pre_trained_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, input_shape=(150,150,3))
pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=(150,150,3),include_top=False)
# pre_trained_model.load_weights('../Week3_2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
for layer in pre_trained_model.layers:
    layer.trainable = False

print(pre_trained_model.summary())
# last_layer = pre_trained_model.get_layer('block6d_add')
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

print(last_layer)
print(last_output)

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(512,activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(3,activation='softmax')(x)

print(pre_trained_model.input)
model = tf.keras.Model(pre_trained_model.input,x)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])

train_dir = 'rps'
valid_dir = 'rps-test-set'

train_datagen = IDG(rescale=1/255.0,rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,
                       zoom_range=0.1,shear_range=0.1,fill_mode='nearest')
val_datagen = IDG(rescale=1/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode='categorical',batch_size=126)
val_generator = val_datagen.flow_from_directory(valid_dir,target_size=(150,150),class_mode='categorical',batch_size=126)

model.fit(train_generator,epochs=15,validation_data=val_generator)

model.save('transferlearning_rps.h5')