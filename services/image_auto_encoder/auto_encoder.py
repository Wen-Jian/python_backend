import tensorflow as tf
import os
from tensorflow.python.framework.ops import disable_eager_execution
import pdb
import glob
import numpy as np
import cv2
import sys
import random
import pdb

disable_eager_execution()

class AutoEncoder:
    def __init__(self, input_shape, filter_shape, activation_function=None, strides=2):
        self.channel_size = input_shape[3]
        batch_size = input_shape[0]
        self.x_batch = tf.compat.v1.placeholder(
            tf.float32, [batch_size, input_shape[1], input_shape[2], self.channel_size], "x_train")

        self.layer_1 = self.add_cnn_layer(
            self.x_batch, filter_shape, activation_function, strides)

        self.layer_2 = self.add_cnn_layer(
            self.layer_1, [filter_shape[0], filter_shape[1], filter_shape[2], 32], activation_function, strides)

        self.layer_3 = self.add_cnn_layer(
            self.layer_2, [filter_shape[0], filter_shape[1], 32, 64], activation_function, strides)

        # self.pooling = self.add_pooling_layer(self.layer_3)
        # self.unpolling = self.add_unpooling_layer(self.pooling, self.layer_3.shape)

        # self.layer_4 = self.add_deconv_layer(self.unpolling, filter_shape, self.layer_2.shape)

        self.layer_4 = self.add_deconv_layer(self.layer_3, [filter_shape[0], filter_shape[1], 32, 64], self.layer_2.shape)

        self.layer_5 = self.add_deconv_layer(self.layer_4, [filter_shape[0], filter_shape[1], filter_shape[2], 32], self.layer_1.shape)

        self.layer_6 = self.add_deconv_layer(self.layer_5, filter_shape, self.x_batch.shape)


        self.loss = tf.reduce_mean(tf.square(self.layer_6 - self.x_batch))
                
        self.optimizer = tf.compat.v1.train.AdamOptimizer(0.0001)

        self.train_step = self.optimizer.minimize(self.loss)

        self.saver = tf.compat.v1.train.Saver()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        self.sess = tf.compat.v1.Session()

        if os.path.isfile("trained_parameters/auto_2.index"):
            self.saver.restore(sess=self.sess, save_path="trained_parameters/auto_2")
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self, imgs_src):
        if type(imgs_src) is str:
            datasets = self.img_to_data_set(imgs_src)
        else:
            datasets = imgs_src
        save_path = "trained_parameters/auto_2"

        try:
            # while True:
            #     _, loss_val = self.sess.run([self.train_step, self.loss], feed_dict={self.x_batch: datasets})
            #     self.saver.save(sess=self.sess, save_path=save_path)
            #     print(loss_val)
            _, loss_val = self.sess.run([self.train_step, self.loss], feed_dict={self.x_batch: datasets})
            self.saver.save(sess=self.sess, save_path=save_path)
            print(loss_val)

        except:
            raise

    def add_cnn_layer(self, x_input, filter_shape, activation_function=None, strides=1):
        cnn_filter = tf.Variable(tf.random.truncated_normal(filter_shape, dtype=tf.dtypes.float32))
        bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        before_pooling = tf.nn.conv2d(x_input, cnn_filter, strides=[
                                      1, strides, strides, 1], padding='SAME')
        if (activation_function != None):
            act_input = activation_function(before_pooling)
        else:
            act_input = tf.nn.relu(before_pooling + bias)
        return act_input

    def add_deconv_layer(self, x_input, filter_shape, output_shape, activation_function=None):
        cnn_filter = tf.Variable(tf.random.truncated_normal(filter_shape, dtype=tf.dtypes.float32))
        bias = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        output_shape = tf.stack(output_shape)
        y1 = tf.nn.conv2d_transpose(
            x_input, cnn_filter, output_shape, strides=[1, 2, 2, 1])
        if (activation_function == None):
            out_put = tf.nn.relu(y1)
        else:
            out_put = activation_function(y1 + bias)
        return out_put

    def add_pooling_layer(self, tensor, stride=2):
        pooling = tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1], strides=[
                                 1, stride, stride, 1], padding='SAME')
        return pooling

    def add_unpooling_layer(self, x, output_shape):
        out = tf.concat([x, tf.zeros_like(x)], 3)
        out = tf.concat([out, tf.zeros_like(out)], 2)
        out_size = output_shape
        return tf.reshape(out, out_size)

    def img_to_data_set(self, files_path = None):
        if files_path == None:
            files_path = os.path.join(os.getcwd(),'MnistImage/Train')
        
        # filelist = glob.glob(os.path.join(files_path, '*'))
        filelist = os.listdir(files_path)
        filelist.sort()
        images = []
        count = 0
        for file_name in filelist:
            process_rate = float(count/len(filelist)) * 100
            temp_img = self.parse_function(files_path + '\\' + file_name)
            if np.shape(temp_img) == (1080, 1920, 3):
                images.append(temp_img) 
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return images

    def img_to_data_set_batch(self, files_path, batch_size):
        # filelist = glob.glob(os.path.join(files_path, '*'))
        filelist = os.listdir(files_path)
        end_index = len(filelist) -1
        index_start = random.randint(0,end_index)
        index_end = index_start + batch_size
        if index_end > end_index:
            index_end = index_start - batch_size
        # name_list = filelist[index_start:index_end]
        name_list = filelist[0:1]
        images = []
        count = 0
        for file_name in name_list:
            process_rate = float(count/len(name_list)) * 100
            temp_img = self.parse_function(files_path + '\\' + file_name)
            if np.shape(temp_img) == (108, 192, 3):
                images.append(temp_img) 
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return images
    
    def parse_function(self, filename):
        os.path.join(os.getcwd(), filename)
        path = filename.replace('./', '\\')
        path = path.replace('/', '\\')
        path = os.getcwd() + path
        image_np_array = cv2.imread(path)
        return image_np_array

    def generate(self, img):
        output = self.sess.run(self.layer_6, feed_dict={self.x_batch: img})
        x_batch = self.sess.run(self.x_batch, feed_dict={self.x_batch: img})
        cv2.imshow('output', np.array(output[0]).astype(np.uint8))
        cv2.imshow('image', img[0])
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
