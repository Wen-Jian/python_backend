import cv2
import numpy as np
import pdb
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from wgan.wgan import Wgan
# from image_auto_encoder.auto_encoder import AutoEncoder
import tensorflow as tf

batch_size = 1

# encoder = AutoEncoder([batch_size, 1080, 1920, 3], [3,3,3,3])
# train_x_file_path = "./app/assets/images/x"
# data_set = encoder.img_to_data_set(train_x_file_path)
# count = 0

# while True:
#     index = (count % 10) * batch_size
#     data = data_set[index:index+batch_size]
#     encoder.train(data)
#     if count == 22:
#         count = 0
#     else:
#         count += 1

# train_x_file_path = "./app/assets/images/x2"
# while True:
#     data_set = encoder.img_to_data_set_batch(train_x_file_path, batch_size)
#     if len(data_set) != batch_size:
#         continue
#     encoder.train(data_set)

# batch_size = 1
# encoder = AutoEncoder([batch_size, 1080, 1920, 3], [3,3,3,3])
# train_x_file_path = "./app/assets/images/y"
# data_set = encoder.img_to_data_set(train_x_file_path)
# encoder.generate(data_set)



# train_x_file_path = "./app/assets/images/test"
# count = 1

(train_x, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
train_x = (train_x - 127.5) / 127.5
data_shape = np.shape(train_x)
model = Wgan([data_shape[1], data_shape[2], data_shape[3]], 512)
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_x)
    .shuffle(60000)
    .batch(512, drop_remainder=True)
    .repeat()
)
while True:
    # data_set = encoder.img_to_data_set_batch(train_x_file_path, batch_size)
    # if len(data_set) != batch_size:
    #     continue
    # model.train(data_set, count % 20 == 0)
    # count += 1
    model.train(train_ds)
