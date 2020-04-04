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
Round = 60
Clinets_per_round = 10
Batch_size = 3000
Test_accuracy = []
Target_accuracy = []
Others_accuracy = []
Models = { }
Attack_models = { }
Client_data = {}
Client_labels = {}


# Data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalization
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5   # Normalization
# 每个用户拥有3种类型的数据，攻击者并没有其他类别的样本
# attack_label = 0
# attack_images = train_images[train_labels == attack_label]
# train_images = np.delete(train_images, train_images==attack_label)
# train_labels = np.delete(train_labels, train_images==attack_label)
for i in range(10):
    Client_data.update({i:np.vstack((np.vstack((train_images[train_labels==i], train_images[train_labels==(i+1)%9])), train_images[train_labels==(i+2)%9]))})
    Client_labels.update({i:np.append(np.append(train_labels[train_labels==i], train_labels[train_labels==(i+1)%9]), train_labels[train_labels==(i+2)%9])})
    print(len(train_labels[train_labels==i]))



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
        ideal_result[i] = 0
    
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
        

        # real output取的是你要模仿的那个数字的概率！！！
        real_output = malicious_discriminator(images, training=False)
        
        fake_output = malicious_discriminator(generated_images, training=False)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, real_labels = labels)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, malicious_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, malicious_discriminator.trainable_variables))




# Train
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, validation_split=0, epochs=1, batch_size = 10000)
del train_images, train_labels

# federated learning
for r in range(Round):
    print('round:'+str(r))
    model_weights_sum=[]

       
    for i in range(Clinets_per_round):
        # if i != 0:
        #     break

        # train the clients individually
        if r != 0:
            Models[i].set_weights(tmp_weight)

               
        train_ds = Client_data[i]
        train_l = Client_labels[i]
        Models[i].fit(train_ds, train_l, validation_split=0, epochs=1, batch_size = Batch_size)     
        if i == 0:
            model_weights_sum = np.array(Models[i].get_weights())
            # print(type(Models[i].get_weights()))
            # print(model_weights_sum)
        else:
            model_weights_sum += np.array(Models[i].get_weights())
            # print(model_weights_sum)


    # averaging the weights
    mean_weight = np.true_divide(model_weights_sum,Clinets_per_round)
    tmp_weight = mean_weight.tolist()
    del model_weights_sum

    # evaluate
    model.set_weights(tmp_weight)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    Test_accuracy.append(test_acc)
    print('\nTest accuracy:', test_acc, 'Tset loss:', test_loss)

    

   


# figures
x=np.arange(0,55)
y=np.array(Test_accuracy)
y1=np.array(Target_accuracy)
plt.title("Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Acc")
plt.plot(x,y)
plt.plot(x,y1)
plt.show()
plt.savefig("fedrated_learning_atack.png")