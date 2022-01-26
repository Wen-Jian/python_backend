import tensorflow as tf

from tensorflow.keras import layers
import cv2
import os
import numpy as np
import pdb
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# tf.compat.v1.enable_eager_execution()
class Autoencoder:
    ITERATION = 10000
    Z_DIM = 100
    D_LR = 0.0000000004
    G_LR = 0.0000000004
    RANDOM_SEED = 42

    def __init__(self, input_shape, batch_size):
        self.checkpoint_path = 'trained_parameters/autoencoder/3x3_512/'
        self.batch_size = batch_size
        self.input_shape = input_shape
        np.random.seed(self.RANDOM_SEED)
        tf.random.set_seed(self.RANDOM_SEED)
        self.generator = self.make_generator((self.Z_DIM,), (batch_size, input_shape[0], input_shape[1], input_shape[2]))
        if os.path.isfile(self.checkpoint_path + "generator/ckpt_generator.index"):
            self.generator.load_weights(self.checkpoint_path + "generator/ckpt_generator")
        
        # optimizer
        self.g_optim = tf.keras.optimizers.Adam(self.G_LR, beta_1=0.5, beta_2=0.999)
        self.test_imgs = None

    def make_generator(self, input_shape, output_shape):
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

    def auto_encoder_loss_fn(self, real_logits, fake_logits):
        return tf.reduce_mean(tf.square(fake_logits - real_logits))

    
    @tf.function
    def train_step(self, real_images):
        with tf.device("/gpu:0"):
            with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                fake_images = self.generator(real_images, training=True)

                g_loss = self.auto_encoder_loss_fn(tf.cast(real_images, tf.float32), fake_images)

            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optim.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss
    
    def train(self, ds, log=False, reduce_learning_rate=False):
        ds = iter(ds)
        images = next(ds)
        if (self.test_imgs == None):
            self.test_imgs = images
        encoder_loss = self.train_step(images)
        self.generator.save_weights(self.checkpoint_path + "generator/ckpt_generator")
        print('encoder_loss: {}'.format(encoder_loss))
        if (log):
            cv2.imshow('input', np.array(self.test_imgs * 127.5 + 127.5).astype(np.uint8)[len(images) - 1])
            cv2.imshow('result', np.array(self.generator(self.test_imgs, training=True) * 127.5 + 127.5).astype(np.uint8)[len(images) - 1])
            cv2.waitKey(1)

    def generate(self, ds):
        ds = iter(ds)
        images = next(ds)
        print("Before:", datetime.now().strftime("%H:%M:%S"))
        cv2.imshow('result', np.array(self.generator(images, training=True) * 127.5 + 127.5).astype(np.uint8)[len(images) - 1])
        print("After:", datetime.now().strftime("%H:%M:%S"))
        cv2.waitKey(0)

    def encoder(self):
        return tf.keras.Sequential([
            self.generator.get_layer(index=0),
            self.generator.get_layer(index=1),
            self.generator.get_layer(index=2),
            self.generator.get_layer(index=3),
            self.generator.get_layer(index=4),
            self.generator.get_layer(index=5),
            self.generator.get_layer(index=6),
            self.generator.get_layer(index=7),
            self.generator.get_layer(index=8),
        ])

    def decoder(self):
        return tf.keras.Sequential([
            self.generator.get_layer(index=9),
            self.generator.get_layer(index=10),
            self.generator.get_layer(index=11),
            self.generator.get_layer(index=12),
            self.generator.get_layer(index=13),
            self.generator.get_layer(index=14),
            self.generator.get_layer(index=15),
        ])
