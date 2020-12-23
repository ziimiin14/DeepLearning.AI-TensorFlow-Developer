import numpy as np
import tensorflow as tf


# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>DESIRED_ACCURACY):
                print('\n It reaches 0.999 accuracy!')
                self.model.stop_training = True


    callbacks=myCallback()

    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)),
                                        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                        # Second convo
                                        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                                        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                        # Third convo
                                        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
                                        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                                        # Dense
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(units=512,activation='relu'),
                                        tf.keras.layers.Dense(units=1,activation='sigmoid')])


    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])


    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    # from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
    from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

    train_datagen = IDG(rescale=1/255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(directory='data_set_exercise',
                                                        target_size=(150,150),
                                                        batch_size=10,
                                                        class_mode='binary')

    # train_generator = train_datagen.flow_from_directory(
    # Your Code Here)
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit(train_generator,steps_per_epoch=8,epochs=20,callbacks=[callbacks])


    return history.history['accuracy'][-1]

a=train_happy_sad_model()
print(a)