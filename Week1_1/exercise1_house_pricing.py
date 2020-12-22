import tensorflow as tf
import numpy as np

def house_price(y_new):
    xs = np.array([1,2,3,4,5,6])
    ys = np.array([1,1.5,2,2.5,3,3.5])
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs,ys,epochs=1000)

    return model.predict(y_new)[0]

prediction = house_price([7])
print(prediction)