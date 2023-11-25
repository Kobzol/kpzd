# https://www.tensorflow.org/tutorials/generative/dcgan

import time

import PIL.Image
import cv2
import numpy as np
import tensorflow as tf

from models import make_discriminator_model, make_generator_model
from utils import generate_image_grid, generate_noise

# Dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(np.float32)
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
    BUFFER_SIZE,
    reshuffle_each_iteration=True
).batch(BATCH_SIZE)

seed = generate_noise(16)

# Models
generator = make_generator_model()
generator.compile()
discriminator = make_discriminator_model()
discriminator.compile()

# Optimizers
lr = 0.0001
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)


# Losses
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = generate_noise(BATCH_SIZE)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))
    return (gen_loss, disc_loss)


# Training
epochs = 50

images = []
for epoch in range(epochs):
    start = time.time()

    gen_losses = []
    disc_losses = []

    for (batch_index, image_batch) in enumerate(dataset):
        gen_loss, disc_loss = train_step(image_batch)
        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)

        if batch_index % 20 == 0:
            image = generate_image_grid(generator, seed)
            image = np.array(PIL.Image.open(image).convert("RGB"))
            cv2.imshow("Test", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    print("Gen loss", np.mean(gen_losses), "Disc loss", np.mean(disc_losses))

    image = generate_image_grid(generator, seed)
    images.append(image)

    print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
    tf.keras.models.save_model(generator, "generator.hdf5")
