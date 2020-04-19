#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import glob
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
# from keras.utils import to_categorical
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


# In[2]:


train_datagen = image.ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory('.', classes=['anime_face'], target_size = (128, 128), 
                                              batch_size = 32, color_mode='rgb')


# In[3]:


# Global Constants
NOISE = (1,1,100)
IMAGE_SHAPE = train_gen.image_shape
# GAN_STEPS = int(140000 / train_gen.batch_size)
GAN_STEPS = 300
BATCH_SIZE = 32


# In[4]:


def generator_model(noise=NOISE):
    gen_input = Input(shape=noise)
    generator = layers.Conv2D(filters=1024, kernel_size=5, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(gen_input)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=512, kernel_size=5, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2D(filters=128, kernel_size=4, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2,2), padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=3, kernel_size=5, strides=(2,2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(generator)
#     generator = layers.ReLU()(generator)
    
    model = Model(inputs=gen_input, outputs=generator)
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.5), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[5]:


def discriminator_model(image_shape=IMAGE_SHAPE):
    disc_input = Input(shape=image_shape)
    discriminator = layers.Conv2D(filters=64, kernel_size=5, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(disc_input)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=5, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(disc_input)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=5, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(disc_input)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    
    discriminator = layers.Conv2D(filters=256, kernel_size=5, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(disc_input)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    
    discriminator = layers.Conv2D(filters=256, kernel_size=5, padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(discriminator)
    discriminator = layers.BatchNormalization(momentum=0.5)(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dense(256, activation='relu')(discriminator)
    discriminator = layers.Dense(1, activation='sigmoid')(discriminator)
    
    model = Model(inputs=disc_input, outputs=discriminator)
    model.compile(optimizer=Adam(lr=1e-3, beta_1=0.5), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[6]:


gen_model = generator_model(NOISE)
gen_model.summary()


# In[7]:


disc_model = discriminator_model()
disc_model.summary()
disc_model.trainable = False


# In[8]:


gan_gen_input = Input(shape=NOISE)
gan_gen = gen_model(gan_gen_input)
gan_dis = disc_model(gan_gen)

gan_model = Model(inputs=gan_gen_input, outputs=gan_dis)
gan_model.compile(optimizer=Adam(lr=1e-4, beta_1=0.5), loss=losses.binary_crossentropy, metrics=['accuracy'])
gan_model.summary()


# In[9]:


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


# In[10]:


for file in glob.glob('./GeneratedFigures/*'):
    if file.endswith('.jpg'):
        os.remove(file)
for file in glob.glob('./GANModels/*'):
    if file.endswith('.h5'):
        os.remove(file)


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
    combined_labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))])
    encoder = LabelEncoder()
    combined_labels = encoder.fit_transform(combined_labels)
    
    # Train Discriminator
    disc_model.trainable = True
    gen_model.trainable = False
    print("Training Discriminator")
    disc_metrics = disc_model.train_on_batch(combined_data, combined_labels)
    print("disc_metrics")
    print(disc_metrics)

#     # Merge real and fake data
#     real_faces, real_labels = train_gen.next()
#     print("Batch Index: ", train_gen.batch_index)
#     fake_labels = np.zeros((BATCH_SIZE, 1))
    
#     encoder = LabelEncoder()
#     real_labels = encoder.fit_transform(real_labels)
#     fake_labels = encoder.fit_transform(fake_labels)
    
    
#     # Train Discriminator
#     disc_model.trainable = True
#     gen_model.trainable = False
#     print("Training Real Discriminator")
#     disc_metrics = disc_model.train_on_batch(real_faces, real_labels)
#     print("disc_metrics")
#     print(disc_metrics)
#     print("Training Fake Discriminator")
#     disc_metrics = disc_model.train_on_batch(created_faces, fake_labels)
#     print("disc_metrics")
#     print(disc_metrics)


    
    # Train GAN
    gen_model.trainable = True
    disc_model.trainable = False
    print("Creating GAN Noise")
    gan_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
    gan_labels = np.ones((BATCH_SIZE, 1))
    gan_labels = encoder.fit_transform(gan_labels)
    print("Training GAN")
    gan_metrics = gan_model.train_on_batch(gan_noise, gan_labels)
    print("gan_metrics")
    print(gan_metrics)
    
    if ((step % 50) == 0):
        gan_model.save('./GANModels/model_'+current_time+".h5")


# In[ ]:




