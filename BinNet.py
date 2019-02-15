'''import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)'''


import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

classes = 10

data = np.random.random((10, 32))
labels = np.eye(10, classes)#np.random.random((10, classes))
#labels = np.ones((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(classes, activation='softmax', name='ctg_out_1')(x)
predictions1 = layers.Dense(classes, activation='softmax', name='ctg_out_2')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()              
# Trains for 5 epochs
model.fit(data, labels, batch_size=50, epochs=5000)

'''
model = tf.keras.Model(inputs=inputs, outputs=[predictions, predictions1])

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss={'ctg_out_1':'categorical_crossentropy',
              'ctg_out_2':'categorical_crossentropy'},
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, [labels, labels], batch_size=32, epochs=5000)
'''





