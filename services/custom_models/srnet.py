import tensorflow as tf

from tensorflow.keras import layers
import cv2
import os
import numpy as np
import pdb
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Srnet:
    def __init__(self, input_shape, batch_size):
        self.checkpoint_path = 'trained_parameters/autoencoder/3x3_512_srnet/'
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = (input_shape[0] *2, input_shape[1] *2, input_shape[2])
        output_batch_shape = (batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2])
        self.generator = self.make_upscaling_generator(output_batch_shape) if upsampling else self.make_generator((self.Z_DIM,), output_batch_shape)
        if os.path.isfile(self.checkpoint_path + "generator/ckpt_generator.index"):
            self.generator.load_weights(self.checkpoint_path + "generator/ckpt_generator")
        
        # optimizer
        self.g_optim = tf.keras.optimizers.Adam(self.G_LR, beta_1=0.5, beta_2=0.999)
        
        self.test_imgs = None

    def make_generator(self, output_shape):
        return tf.keras.Sequential([
            # for image to image
            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                512, 3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                256, 3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                128, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                output_shape[3], 3, strides=2, padding='same', use_bias=False, activation='tanh')
        ])