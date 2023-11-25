import tensorflow as tf


def make_generator_model() -> tf.keras.models.Model:
    input = tf.keras.layers.Input(shape=(100,))
    layer = tf.keras.layers.Dense(7 * 7 * 256, use_bias=False)(input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Reshape((7, 7, 256))(layer)
    assert layer.shape == (None, 7, 7, 256)  # Note: None is the batch size

    layer = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                                            padding="same", use_bias=False)(layer)
    assert layer.shape == (None, 7, 7, 128)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)

    layer = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same",
                                            use_bias=False)(layer)
    assert layer.shape == (None, 14, 14, 64)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)

    layer = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same",
                                            use_bias=False, activation="tanh")(layer)
    assert layer.shape == (None, 28, 28, 1)

    return tf.keras.models.Model(inputs=[input], outputs=[layer])


def make_discriminator_model() -> tf.keras.models.Model:
    input = tf.keras.layers.Input(shape=(28, 28, 1))

    layer = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same")(input)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Dropout(0.3)(layer)

    layer = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(layer)
    layer = tf.keras.layers.LeakyReLU()(layer)
    layer = tf.keras.layers.Dropout(0.3)(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(1)(layer)

    return tf.keras.models.Model(inputs=[input], outputs=[layer])
