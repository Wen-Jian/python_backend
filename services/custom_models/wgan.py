import tensorflow as tf

from tensorflow.keras import layers
import cv2
import os
import numpy as np
import pdb

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Wgan:
    ITERATION = 10000
    Z_DIM = 100
    D_LR = 0.000004
    G_LR = 0.000004
    RANDOM_SEED = 42

    def __init__(self, input_shape, batch_size):
        self.checkpoint_path = 'trained_parameters/wgan/5x5_512/'
        self.batch_size = batch_size
        np.random.seed(self.RANDOM_SEED)
        tf.random.set_seed(self.RANDOM_SEED)
        self.generator = self.make_generator((self.Z_DIM,), (batch_size, input_shape[0], input_shape[1], input_shape[2]))
        if os.path.isfile(self.checkpoint_path + "generator/ckpt_generator.index"):
            self.generator.load_weights(self.checkpoint_path + "generator/ckpt_generator")
        self.discriminaor = self.make_discriminaor(input_shape)
        if os.path.isfile(self.checkpoint_path + "discriminaor/ckpt_discriminaor.index"):
            self.discriminaor.load_weights(self.checkpoint_path + "discriminaor/ckpt_discriminaor")
        
        # optimizer
        self.g_optim = tf.keras.optimizers.Adam(self.G_LR, beta_1=0.5, beta_2=0.999)
        self.d_optim = tf.keras.optimizers.Adam(self.D_LR, beta_1=0.5, beta_2=0.999)
        self.test_imgs = None

    def make_generator(self, input_shape, output_shape):
        width = int(output_shape[1]/4)
        height = int(output_shape[2]/4)

        return tf.keras.Sequential([
            # for random seed generator
            layers.Dense(width*height*512, use_bias=False, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((width, height, 512)),
            layers.Conv2DTranspose(
                256, 3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                128, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                3, 3, strides=2, padding='same', use_bias=False, activation='tanh')
        ])

    def make_discriminaor(self, input_shape):
        return tf.keras.Sequential([
            layers.Conv2D(64, 5, strides=2, padding='same',
                        input_shape=input_shape),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, 5, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def d_loss_fn(self, real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
        # """The discriminator loss function."""
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # return bce(tf.ones_like(real_logits), real_logits) + bce(
        #     tf.zeros_like(fake_logits), fake_logits
        # )

    def g_loss_fn(self, fake_logits):
        return -tf.reduce_mean(fake_logits)
        # """The Generator loss function."""
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        # return bce(tf.ones_like(fake_logits), fake_logits)

    @tf.function
    def train_step(self, real_images):
        z = self.get_random_z(self.Z_DIM, self.batch_size)
        with tf.device("/gpu:0"):
            with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
                fake_images = self.generator(z, training=True)
                fake_logits = self.discriminaor(fake_images, training=True)
                real_logits = self.discriminaor(real_images, training=True)

                d_loss = self.d_loss_fn(real_logits, fake_logits)
                g_loss = self.g_loss_fn(fake_logits)

            d_gradients = d_tape.gradient(d_loss, self.discriminaor.trainable_variables)
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

            self.d_optim.apply_gradients(zip(d_gradients, self.discriminaor.trainable_variables))
            self.g_optim.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss, d_loss

    def get_random_z(self, z_dim, batch_size):
        return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)

    def train(self, ds, log=False, reduce_learning_rate=False):
        test_z = self.get_random_z(self.Z_DIM, self.batch_size)
        ds = iter(ds)
        images = next(ds)
        if (self.test_imgs == None):
            self.test_imgs = images
        if (reduce_learning_rate):
            self.D_LR = self.D_LR / 10
            self.G_LR = self.G_LR / 10
        g_loss, d_loss = self.train_step(images)
        self.generator.save_weights(self.checkpoint_path + "generator/ckpt_generator")
        self.discriminaor.save_weights(self.checkpoint_path + "discriminaor/ckpt_discriminaor")
        print('d_loss: {}'.format(d_loss))
        print('g_loss: {}'.format(g_loss))
        if (log):
            # cv2.imshow('input', np.array(self.test_imgs * 127.5 + 127.5).astype(np.uint8)[len(images) - 1])
            cv2.imshow('result', np.array(self.generator(test_z, training=True) * 127.5 + 127.5).astype(np.uint8)[len(images) - 1])
            cv2.waitKey(1)