import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# image_datagen = IDG(validation_split=0.2,rescale=1/255.0)
#
# train_generator = image_datagen.flow_from_directory('dog_cat_dataset',subset='training')
# val_generator = image_datagen.flow_from_directory('dog_cat_dataset',subset='validation')
#
# print(train_generator)
# print(val_generator)
train_data_dir = 'dog_cat_classes'
batch_size = 100
img_height = 150
img_width = 150

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation') # set as validation data