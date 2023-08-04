import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
# from dcgan import make_generator_model

import os
import numpy as np
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu)

noise_dim = 100
resultsDir = './results_sky/WSISEG256/img_distributions/'
model_dir = './models_sky/bestCKPT_WSISEG256/'

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 256, 256, 3)

    return model

generator = make_generator_model()


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(4, 4), padding='same',
                                     input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(4, 4), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
print("Model Loaded")

print("Results Directory", resultsDir)
if not os.path.isdir(resultsDir):
    os.mkdir(resultsDir)
for directory in os.listdir(resultsDir):
    if os.path.isdir(directory):
        os.rmdir(directory)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


print("Generating Samples")
for idx in range(20):
    checkpoint.restore(model_dir + f'ckpt-{idx+1}')
    print("Checkpoint Restored", model_dir + f'ckpt-{idx+1}')

    epoch_path = resultsDir+f'epoch{250*(idx+1)}/'
    Path(epoch_path).mkdir(parents=True, exist_ok=True)

    noise = tf.random.normal([100, noise_dim])
    predictions = generator(noise, training=False)
    gen_images = (predictions.numpy()*255/2) + 255/2
    gen_images = gen_images.astype(np.uint8)

    for i in tqdm(range(predictions.shape[0])):       
        im = Image.fromarray(gen_images[i, :, :, :])
        im.save(epoch_path + f"sample{i}.png")

