
from statistics import mode
import numpy as np
import pdb
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from autoencoder.autoencoder import Autoencoder
from image_loader import ImageLoader
# from image_auto_encoder.auto_encoder import AutoEncoder
import tensorflow as tf
import cv2

batch_size = 32

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



# train_x_file_path = "./app/assets/images/small_x"
# count = 1
# # train_x = np.array(ImageLoader.img_to_data_set(train_x_file_path))
# # # (train_x, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
# # train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
# # train_x = (train_x - 127.5) / 127.5
# # data_shape = np.shape(train_x)
# train_x = np.array(ImageLoader.img_to_data_set_batch(train_x_file_path, batch_size))
# train_x = (train_x - 127.5) / 127.5
# data_shape = np.shape(train_x)
# model = Autoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size)

# while True:
    
#     train_ds = (
#         tf.data.Dataset.from_tensor_slices(train_x)
#         .shuffle(buffer_size=batch_size)
#         .batch(batch_size)
#     )
#     # data_set = encoder.img_to_data_set_batch(train_x_file_path, batch_size)
#     if len(train_x) == batch_size:
#         model.train(train_ds, count % 20 == 0, count % 200 == 0)
#         # model.train_auto_encoder(train_ds, count % 20 == 0)

#     train_x = np.array(ImageLoader.img_to_data_set_batch(train_x_file_path, batch_size))
#     train_x = (train_x - 127.5) / 127.5
#     count += 1

train_x_file_path = "./app/assets/images/y/0001.jpg"
batch_size=1
train_x = np.array([ImageLoader.parse_function(train_x_file_path)])
train_x = (train_x - 127.5) / 127.5
data_shape = np.shape(train_x)
print(data_shape)
model = Autoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size)
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_x)
    .shuffle(buffer_size=batch_size)
    .batch(batch_size)
)
ds = iter(train_ds)
images = next(ds)
encoder = model.encoder()
decoder = model.decoder()
output = encoder(images)
result = decoder(output)
cv2.imshow('result', np.array(result * 127.5 + 127.5).astype(np.uint8)[0])
cv2.waitKey(0)