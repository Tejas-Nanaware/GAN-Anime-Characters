#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import glob
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.utils import plot_model
from PIL import Image
from keras import Input
from keras import layers
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras import Model
from keras import losses

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# MacOS matplotlib kernel issue
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[2]:


# Global Constants
NOISE = (100,)
IMAGE_SHAPE = (64,64,3)
GAN_STEPS = 500
BATCH_SIZE = 64


# In[3]:


def generator_model(noise=NOISE):
    gen_input = Input(shape=noise)
    
    generator = layers.Dense(32 * 32 * 128, use_bias=False)(gen_input)
    generator = layers.LeakyReLU()(generator)
    generator = layers.Reshape((32, 32, 128))(generator)
    
    generator = layers.Conv2D(filters=512, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU()(generator)
    
    generator = layers.Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU()(generator)
    
    generator = layers.Conv2DTranspose(filters=256, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU()(generator)
    
    generator = layers.Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU()(generator)
    
    generator = layers.Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(1,1), use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    generator = layers.BatchNormalization(momentum=0.5)(generator)
    generator = layers.LeakyReLU()(generator)
        
    generator = layers.Conv2DTranspose(filters=3, kernel_size=(7,7), strides=(1,1), activation='tanh', use_bias=False, padding='same', kernel_initializer='glorot_uniform')(generator)
    
    model = Model(inputs=gen_input, outputs=generator)
#     model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[4]:


def discriminator_model(image_shape=IMAGE_SHAPE):
    disc_input = Input(shape=image_shape)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3))(disc_input)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=(4,4), padding='same', strides=(2,2), kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3))(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=(4,4), padding='same', strides=(2,2), kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3))(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)
    
    discriminator = layers.Conv2D(filters=128, kernel_size=(4,4), padding='same', strides=(2,2), kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3))(discriminator)
    discriminator = layers.LeakyReLU()(discriminator)
    discriminator = layers.Dropout(0.3)(discriminator)
    
    discriminator = layers.Flatten()(discriminator)
    discriminator = layers.Dropout(0.4)(discriminator)
    discriminator = layers.Dense(1, activation='sigmoid')(discriminator)
    
    model = Model(inputs=disc_input, outputs=discriminator)
    model.compile(optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8), loss=losses.binary_crossentropy, metrics=['accuracy'])
    
    return model


# In[5]:


gen_model = generator_model(NOISE)
gen_model.summary()


# In[6]:


disc_model = discriminator_model()
disc_model.summary()
disc_model.trainable = False


# In[7]:


gan_gen_input = Input(shape=NOISE)
gan_gen = gen_model(gan_gen_input)
gan_dis = disc_model(gan_gen)

gan_model = Model(inputs=gan_gen_input, outputs=gan_dis)
gan_model.compile(optimizer=RMSprop(lr=0.0002, clipvalue=1.0, decay=1e-8), loss=losses.binary_crossentropy, metrics=['accuracy'])
gan_model.summary()


# plot_model(gen_model, to_file='./PlotsAndImages/gen_model_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(disc_model, to_file='./PlotsAndImages/disc_model_plot.png', show_shapes=True, show_layer_names=True)
# plot_model(gan_model, to_file='./PlotsAndImages/gan_model_plot.png', show_shapes=True, show_layer_names=True)

# In[8]:


def save_fig(predicted, current_time):
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
#         my_image = (predicted[i] * 255).astype(np.uint8) / 255
        plt.imshow(my_image)
    plt.savefig('./GeneratedFigures/image_'+current_time+'.jpg', bbox_inches = 'tight', pad_inches = 0.1)
#     plt.show(block=True)
    plt.close('all')


# In[9]:


def PIL_get_images(batch_size=BATCH_SIZE, image_shape=IMAGE_SHAPE, data_search_pattern='./anime_face/000*'):
    batch_image_shape = (batch_size,) + image_shape
    batch_images = np.empty(batch_image_shape, dtype=np.float32)
    all_images = glob.glob(data_search_pattern)
    chosen_images = np.random.choice(all_images, batch_size)
    for index, file in enumerate(chosen_images):
        # Read images using PIL
        my_img = Image.open(file).resize(IMAGE_SHAPE[:-1])
        my_img = my_img.convert('RGB')
        my_img = np.asarray(my_img, dtype=np.float)
        my_img = (my_img / 127.5) - 1
        batch_images[index,...] = my_img
    return batch_images


# In[10]:


for file in glob.glob('./GeneratedFigures/*'):
    if file.endswith('.jpg'):
        os.remove(file)
for file in glob.glob('./GANModels/*'):
    if file.endswith('.h5'):
        os.remove(file)


# In[11]:


with open('log.csv', 'w') as log:
#     log.write('Step,DiscLoss,DiscAcc,GANLoss,GANAcc\n')
    log.write('Step,RealDiscLoss,RealDiscAcc,GenDiscLoss,GenDiscAcc,GANLoss,GANAcc\n')
for step in range(1, GAN_STEPS+1):
    print('**************************************')
    print()
    print('               Step: ', step)
    print()
    print('**************************************')
    
    current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    current_time = str(step)
    print("Creating noise")
    gen_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
    print("Predicting noise")
    created_faces = gen_model.predict(gen_noise)
    if ((step % 10) == 0):
        save_fig(created_faces, current_time)
    
#     # Merge real and fake data
#     real_faces = PIL_get_images()
# #     print("Batch Index: ", train_gen.batch_index)
#     combined_data = np.concatenate([real_faces, created_faces])
#     combined_labels = np.concatenate([np.ones((BATCH_SIZE, 1), dtype=np.int).ravel(), np.zeros((BATCH_SIZE, 1), dtype=np.int).ravel()])
    
#     # Train Discriminator
#     disc_model.trainable = True
#     gen_model.trainable = False
#     print("Training Discriminator")
#     disc_metrics = disc_model.train_on_batch(combined_data, combined_labels)
#     print("disc_metrics")
#     print(disc_metrics)

    # Merge real and fake data
    real_faces = PIL_get_images()
#     print("Batch Index: ", train_gen.batch_index)
    real_labels = np.ones((BATCH_SIZE, 1), dtype=np.int).ravel()
    fake_labels = np.zeros((BATCH_SIZE, 1), dtype=np.int).ravel()   
    
    # Train Discriminator
    disc_model.trainable = True
    gen_model.trainable = False
    print("Training Real Discriminator")
    real_disc_metrics = disc_model.train_on_batch(real_faces, real_labels)
    print("disc_metrics")
    print(real_disc_metrics)
    print("Training Fake Discriminator")
    gen_disc_metrics = disc_model.train_on_batch(created_faces, fake_labels)
    print("disc_metrics")
    print(gen_disc_metrics)


    
    # Train GAN
    gen_model.trainable = True
    disc_model.trainable = False
    print("Creating GAN Noise")
    gan_noise = np.random.normal(loc=0, scale=1, size=(BATCH_SIZE,)+NOISE)
    gan_labels = np.ones((BATCH_SIZE, 1), dtype=np.int).ravel()
    print("Training GAN")
    gan_metrics = gan_model.train_on_batch(gan_noise, gan_labels)
    print("gan_metrics")
    print(gan_metrics)
    
    # Append Log
    with open('log.csv', 'a') as log:
#         log.write('%d,%f,%f,%f,%f\n' % (step, disc_metrics[0], disc_metrics[1], gan_metrics[0], gan_metrics[1]))
        log.write('%d,%f,%f,%f,%f,%f,%f\n' % (step, real_disc_metrics[0], real_disc_metrics[1], gen_disc_metrics[0], gen_disc_metrics[1], gan_metrics[0], gan_metrics[1]))
    if ((step % 100) == 0):
        gan_model.save('./GANModels/GANmodel_'+current_time+'.h5')
        gen_model.trainable = True
        gen_model.save('./GANModels/GENmodel_'+current_time+'.h5')
        disc_model.trainable = True
        disc_model.save('./GANModels/DISmodel_'+current_time+'.h5')


# In[ ]:




