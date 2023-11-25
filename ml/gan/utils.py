import io

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def generate_noise(count=1) -> tf.Tensor:
    return tf.random.normal([count, 100])


def show_image(image: np.ndarray):
    image = image[:, :, 0]
    image = image * 127.5 + 127.5
    image = image.astype(np.uint8)
    cv2.imshow("Test", image)
    cv2.waitKey(0)


def generate_image_grid(model, test_input) -> io.BytesIO:
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    return buffer
