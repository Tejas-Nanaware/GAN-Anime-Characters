#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import time
import glob
from keras.preprocessing import image
from keras.utils import to_categorical
from keras import Input
from keras import layers
from keras.optimizers import Adam
from keras import Model
from keras import losses

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# MacOS matplotlib kernel issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[3]:


train_datagen = image.ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory('.', classes=['anime_face'], target_size = (128, 128), 
                                              batch_size = 32, color_mode='rgb')


# In[4]:


# Global Constants
NOISE = (128,128,128)
IMAGE_SHAPE = train_gen.image_shape
# GAN_STEPS = int(140000 / train_gen.batch_size)
GAN_STEPS = 1000
BATCH_SIZE = 32


# In[5]:


def generator_model(noise=NOISE):
    gen_input = Input(shape=noise)
    generator = layers.Conv2D(filters=512, kernel_size=5, padding='same', dilation_rate=4)(gen_input)
    generator = layers.LeakyReLU(0.1)(generator)
    generator = layers.Conv2DTranspose(filters=256, kernel_size=5, padding='same', dilation_rate=4)(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.Conv2D(filters=3, kernel_size=5, activation='tanh', padding='same', dilation_rate=4)(generator)
    
    model = Model(inputs=gen_input, outputs=generator)
    model.compile(optimizer=Adam(lr=0.01), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    
    return model


# In[6]:


def discriminator_model(image_shape=IMAGE_SHAPE):
    disc_input = Input(shape=image_shape)
    discriminator = layers.Conv2D(filters=128, kernel_size=5)(disc_input)
    discriminator = layers.LeakyReLU(0.1)(discriminator)
    discriminator = layers.Conv2DTranspose(filters=256, kernel_size=5)(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.Conv2D(filters=512, kernel_size=5)(discriminator)
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dense(2, activation='sigmoid')(discriminator)
    
    model = Model(inputs=disc_input, outputs=discriminator)
    model.compile(optimizer=Adam(lr=0.01), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[7]:


gen_model = generator_model(NOISE)
gen_model.summary()


# In[8]:


disc_model = discriminator_model()
disc_model.summary()
disc_model.trainable = False


# In[9]:


gan_gen_input = Input(shape=NOISE)
gan_gen = gen_model(gan_gen_input)
gan_dis = disc_model(gan_gen)

gan_model = Model(inputs=gan_gen_input, outputs=gan_dis)
gan_model.compile(optimizer=Adam(lr=0.01), loss=losses.binary_crossentropy, metrics=['accuracy'])
gan_model.summary()


# In[13]:


def save_fig(predicted, current_time):
    num_images = predicted.shape[0]
    fig = plt.figure(figsize=(15,7))
    columns = 8
    rows = np.ceil(num_images / columns)
    for i in range(num_images):
        fig.add_subplot(rows, columns, i+1)
        my_image = (predicted[i] * 255).astype(np.uint8) / 255
        plt.imshow(my_image)
    plt.savefig('./GeneratedFigures/image_'+current_time+'.jpg', bbox_inches = 'tight', pad_inches = 0.1)
#     plt.show(block=True)


# In[20]:


for file in glob.glob('./GeneratedFigures/*'):
    if file.endswith('.jpg'):
        os.remove(file)
for file in glob.glob('./GANModels/*'):
    if file.endswith('.h5'):
        os.remove(file)


# In[14]:


gen_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
predicted = gen_model.predict(gen_noise)
current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
save_fig(predicted, current_time)


# In[11]:


for step in range(GAN_STEPS):
    print('**************************************')
    print()
    print('               Step: ', step)
    print()
    print('**************************************')
    
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print("Creating noise")
    gen_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
    print("Predicting noise")
    created_faces = gen_model.predict(gen_noise)
    if ((step % 1) == 0):
        save_fig(created_faces, current_time)
    
    # Merge real and fake data
    real_faces, labels = train_gen.next()
    print("Batch Index: ", train_gen.batch_index)
    combined_data = np.concatenate([real_faces, created_faces])
    combined_labels = to_categorical(np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))]))
    
    # Train Discriminator
    disc_model.trainable = True
    gen_model.trainable = False
    print("Training Discriminator")
    disc_metrics = disc_model.train_on_batch(combined_data, combined_labels)
    print("disc_metrics")
    print(disc_metrics)
    
    # Train GAN
    gen_model.trainable = True
    disc_model.trainable = False
    print("Creating GAN Noise")
    gan_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
    gan_labels = to_categorical(np.ones((BATCH_SIZE, 1)))
    print("Training GAN")
    gan_metrics = gan_model.train_on_batch(gan_noise, gan_labels)
    print("gan_metrics")
    print(gan_metrics)
    
    if ((step % 100) == 0):
        gan_model.save('./GANModels/model_'+current_time+".h5")

