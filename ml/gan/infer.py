import tensorflow as tf

from utils import generate_noise, show_image

generator = tf.keras.models.load_model("generator.hdf5")

while True:
    generated = generator(generate_noise())
    show_image(generated[0].numpy())
