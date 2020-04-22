#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import glob
from keras.datasets import mnist
from keras import Model
from keras import layers
from keras import Input
from keras import losses
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.data import Dataset
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# MacOS matplotlib kernel issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[5]:


(train_images, train_labels), (_, _) = mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5


# In[6]:


BUFFER_SIZE = 60000
BATCH_SIZE = 256
NOISE = (100,)
GAN_STEPS = 1000
IMAGE_SHAPE = (28, 28, 1)


# In[15]:


def generator_model(noise=NOISE):
    gen_input = Input(shape=noise)
    
    generator = layers.Dense(7 * 7 * 64, use_bias=False)(gen_input)
    generator = layers.BatchNormalization(momentum=0.9)(generator)
    enerator = layers.LeakyReLU(alpha=0.2)(generator)
    generator = layers.Reshape((7, 7, 64))(generator)
    generator = layers.Dropout(0.4)(generator)
    
    generator = layers.UpSampling2D()(generator)
    generator = layers.Conv2DTranspose(filters=128, kernel_size=(5,5), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.9)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.UpSampling2D()(generator)
    generator = layers.Conv2DTranspose(filters=64, kernel_size=(5,5), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.9)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
    
    generator = layers.Conv2DTranspose(filters=32, kernel_size=(5,5), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.9)(generator)
    generator = layers.LeakyReLU(alpha=0.2)(generator)
            
    generator = layers.Conv2DTranspose(filters=1, kernel_size=(5,5), activation='tanh', use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    
    model = Model(inputs=gen_input, outputs=generator)
    model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[16]:


def discriminator_model(image_shape=IMAGE_SHAPE):
    disc_input = Input(shape=image_shape)
    
    discriminator = layers.Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(disc_input)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Dropout(0.4)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=(5,5), padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Dropout(0.4)(discriminator)
    
    discriminator = layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', strides=(2,2), kernel_initializer='glorot_uniform')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Dropout(0.4)(discriminator)
    
    discriminator = layers.Conv2D(filters=512, kernel_size=(5,5), padding='same', strides=(1,1), kernel_initializer='glorot_uniform')(discriminator)
    discriminator = layers.LeakyReLU(alpha=0.2)(discriminator)
    discriminator = layers.Dropout(0.4)(discriminator)
        
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dense(1, activation='sigmoid')(discriminator)
    
    model = Model(inputs=disc_input, outputs=discriminator)
    model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[17]:


gen_model = generator_model(NOISE)
gen_model.summary()


# In[18]:


disc_model = discriminator_model()
disc_model.summary()
disc_model.trainable = False


# In[11]:


gan_gen_input = Input(shape=NOISE)
gan_gen = gen_model(gan_gen_input)
gan_dis = disc_model(gan_gen)

gan_model = Model(inputs=gan_gen_input, outputs=gan_dis)
gan_model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=['accuracy'])
gan_model.summary()


# In[9]:


def save_fig(predicted, step):
    # Only 32 images will be printed
    if BATCH_SIZE > 32:
        predicted = predicted[:32]
    num_images = predicted.shape[0]
    fig = plt.figure(figsize=(15,7))
    columns = 8
    rows = np.ceil(num_images / columns)
    for i in range(num_images):
        fig.add_subplot(rows, columns, i+1)
        my_image = predicted[i]
        # Denormalize Image
        my_image = ((my_image + 1) * 127.5) / 255
        plt.imshow(my_image[:, :, 0], cmap='gray')
    plt.savefig('./GeneratedDigits/image_'+step+'.jpg', bbox_inches = 'tight', pad_inches = 0.1)
#     plt.show(block=True)
    plt.close('all')


# In[10]:


for file in glob.glob('./GeneratedDigits/*'):
    if file.endswith('.jpg'):
        os.remove(file)
for file in glob.glob('./DigitModels/*'):
    if file.endswith('.h5'):
        os.remove(file)


# In[11]:


def train_batch(images):
    # Create digits using generator
    gen_noise = np.random.normal(loc=0, scale=1, size=(images.shape[0],)+NOISE)
    created_digits = gen_model.predict(gen_noise)

    # Train Discriminator
    real_labels = np.ones((images.shape[0], 1), dtype=np.int).ravel()
    fake_labels = np.zeros((images.shape[0], 1), dtype=np.int).ravel()

    disc_model.trainable = True
    gen_model.trainable = False

    real_disc_metrics = disc_model.train_on_batch(images, real_labels)
    gen_disc_metrics = disc_model.train_on_batch(created_digits, fake_labels)

    # Train GAN
    gen_model.trainable = True
    disc_model.trainable = False

    gan_noise = np.random.normal(loc=0, scale=1, size=(images.shape[0],)+NOISE)
    gan_labels = np.ones((images.shape[0], 1), dtype=np.int).ravel()
    gan_metrics = gan_model.train_on_batch(gan_noise, gan_labels)
    
    return real_disc_metrics, gen_disc_metrics, gan_metrics


# In[14]:


with open('mnist_log.csv', 'w') as log:
    log.write('Step,RealDiscLoss,RealDiscAcc,GenDiscLoss,GenDiscAcc,GANLoss,GANAcc\n')

for step in range(1, GAN_STEPS+1):
    print('**************************************')
    print()
    print('               Step: ', step)
    print()
    print('**************************************')
    
    if ((step % 10) == 0):
        gen_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
        created_digits = gen_model.predict(gen_noise)
        save_fig(created_digits, str(step))
    
    real_disc_metrics, gen_disc_metrics, gan_metrics = [], [], []
    counter = 1
    start = 0
    end = BATCH_SIZE
    sliced = train_images[start:end]
    while (sliced.shape[0] > 0):
#         print(counter)
        real_disc_metrics, gen_disc_metrics, gan_metrics = train_batch(sliced)
        counter +=1
        start += BATCH_SIZE
        end += BATCH_SIZE
        sliced = train_images[start:end]
    
    # Append Log
    with open('mnist_log.csv', 'a') as log:
        log.write('%d,%f,%f,%f,%f,%f,%f\n' % (step, real_disc_metrics[0], real_disc_metrics[1], gen_disc_metrics[0], gen_disc_metrics[1], gan_metrics[0], gan_metrics[1]))
    if ((step % 200) == 0):
        gan_model.save('./DigitModels/GANmodel_'+str(step)+'.h5')
        gen_model.trainable = True
        gen_model.save('./DigitModels/GENmodel_'+str(step)+'.h5')
        disc_model.trainable = True
        disc_model.save('./DigitModels/DISmodel_'+str(step)+'.h5')


# In[ ]:




