from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# TensorFlow, tf.keras and tensorflow_federated
import tensorflow as tf
from tensorflow import keras
import tensorflow_federated as tff

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time

# tf2.0 不加报错
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Constants
Round = 150
Clinets_per_round = 10
Batch_size = 2048
Gan_epoch = 50
Test_accuracy = []
Models = { }
Client_data = {}
Client_labels = {}

BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16
num_to_merge = 200
seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed_merge = tf.random.normal([num_to_merge, noise_dim])

#########################################################################
##                             Load Data                               ##
#########################################################################

# Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalization
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5   # Normalization

# 每个Client拥有3种类型的数据，Attacker并没有target类别的样本
for i in range(10):
    Client_data.update({i:np.vstack((np.vstack((train_images[train_labels==i], train_images[train_labels==(i+1)%9])), train_images[train_labels==(i+2)%9]))})
    Client_labels.update({i:np.append(np.append(train_labels[train_labels==i], train_labels[train_labels==(i+1)%9]), train_labels[train_labels==(i+2)%9])})
    state = np.random.get_state()
    np.random.shuffle(Client_data[i])
    np.random.set_state(state)
    np.random.shuffle(Client_labels[i])
    # print(len(train_labels[train_labels==i]))


#########################################################################
##                          Models Prepared                            ##
#########################################################################

# Models & malicious discriminator model
def make_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(11))
    return model

# Malicious generator model
def make_generator_model():
    model = keras.Sequential()
    
    model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Batch size is not limited

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# Model
# Sever‘s models
model = make_discriminator_model()

# Clients' models
for i in range(Clinets_per_round):
    Models.update({i:make_discriminator_model()})
    Models[i].compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#########################################################################
##                            Attack setup                             ##
#########################################################################

# Malicious gan
generator = make_generator_model()
malicious_discriminator = make_discriminator_model()


# Cross entropy
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Loss of discriminator
def discriminator_loss(real_output, fake_output, real_labels):
    real_loss = cross_entropy(real_labels, real_output)
    
    fake_result = np.zeros(len(fake_output))
    # Attack label
    for i in range(len(fake_result)):
        fake_result[i] = 10
    fake_loss = cross_entropy(fake_result, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Loss of generator
def generator_loss(fake_output):
    ideal_result = np.zeros(len(fake_output))
    # Attack label
    for i in range(len(ideal_result)):
        # The class which attacker intends to get
        # suppose 4 here
        ideal_result[i] = 4
    
    return cross_entropy(ideal_result, fake_output)

# Adam optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        # real_output取的是要模仿的那个数字的概率
        real_output = malicious_discriminator(images, training=False)
        fake_output = malicious_discriminator(generated_images, training=False)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, real_labels = labels)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))

# Train
def train(dataset, labels, epochs):
    for epoch in range(epochs):
        start = time.time()
        for i in range(round(len(dataset)/BATCH_SIZE)):
            image_batch = dataset[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
            labels_batch = labels[i*BATCH_SIZE:min(len(dataset), (i+1)*BATCH_SIZE)]
            train_step(image_batch, labels_batch)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # 最后一个epoch结束后生成图片并merge到dataset中
    generate_and_save_images(generator, epochs, seed)

# Generate images to check the effect
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


#########################################################################
##                         Federated Learning                          ##
#########################################################################

# Training Prepare
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_split=0, epochs=1, batch_size = 8192)
del train_images, train_labels

# Federated learning
for r in range(Round):
    print('round:'+str(r))
    model_weights_sum=[]

       
    for i in range(Clinets_per_round):
        # train the clients individually
        if r != 0:
            Models[i].set_weights(tmp_weight)
        
        train_ds = Client_data[i]
        train_l = Client_labels[i]
        Models[i].fit(train_ds, train_l, validation_split=0, epochs=1, batch_size = Batch_size)     
        if i == 0:
            model_weights_sum = np.array(Models[i].get_weights())
        else:
            model_weights_sum += np.array(Models[i].get_weights())

        # Attack (suppose client 0 is malicious)
        if r != 0 and i == 0 and Test_accuracy[i-1] > 0.87:
            malicious_discriminator.set_weights(Models[i].get_weights())
            train(train_ds, train_l, Gan_epoch)

            # Merge the malicious images
            predictions = generator(seed_merge, training=False)
            malicious_images = np.array(predictions)
            np.vstack((Client_data[i], malicious_images))
            # Label the malicious images
            malicious_labels = np.array([0]*len(malicious_images))
            np.append(Client_labels[i], malicious_labels)


    # averaging the weights
    mean_weight = np.true_divide(model_weights_sum,Clinets_per_round)
    tmp_weight = mean_weight.tolist()
    del model_weights_sum

    # evaluate
    model.set_weights(tmp_weight)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    Test_accuracy.append(test_acc)
    print('\nTest accuracy:', test_acc, 'Tset loss:', test_loss)
