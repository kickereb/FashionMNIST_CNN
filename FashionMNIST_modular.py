#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, Dense, Flatten
print(tf.__version__)


# In[ ]:


# tf.disable_v2_behavior()


# In[3]:


fashion_mnist = keras.datasets.fashion_mnist
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

# In[4]:

for_hold = trainX.copy()
for i in range(trainX.shape[0]):
    for_hold[i] = cv2.flip(trainX[i], np.random.randint(-1,3))
trainX = for_hold.copy()


# In[5]:


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))



# In[6]:


trainX = trainX/255.0
testX = testX/255.0


# In[7]:


#input tensor initialisation
input_img = keras.layers.Input((28, 28, 1), name = "img")
n_filters = 16


# In[11]:


# def make_model(input_img, n_filters = 16):
c1 = keras.layers.Conv2D(n_filters, (3,3), padding='same', activation='relu', input_shape = (28,28,1))(input_img)
p1 = keras.layers.MaxPooling2D((2,2))(c1)

c2 = keras.layers.Conv2D(n_filters*4, (3,3), padding='same', activation='relu')(p1)
p2 = keras.layers.MaxPooling2D((2,2))(c2)

c3 = keras.layers.Conv2D(n_filters*4, (3,3), padding='same', activation='relu')(p2)
flat = keras.layers.Flatten()
flattened = flat(c3)
d1 = keras.layers.Dense(64, activation='relu')(flattened)
d2 = keras.layers.Dense(10, activation='softmax')(d1)
model = Model(inputs=[input_img], outputs=[d2])
model.compile(optimizer = 'Adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])


# In[12]:


model.summary()


# In[13]:


history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10)


# In[14]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(trainX, trainY, verbose=2)


# In[ ]:




