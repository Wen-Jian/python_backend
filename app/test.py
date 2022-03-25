
from gc import callbacks
from statistics import mode
import numpy as np
import pdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from custom_models.autoencoder import ContextAutoencoder
from custom_models.autoencoder import Autoencoder
from custom_models.P_COVNET import InpaintingModel, ValidationCallback, InpaintingModelV2, ValidationCallbackV2
from image_processor import ImageProcessor
from image_processor import ImageMasker
import tensorflow as tf
import cv2

batch_size = 2

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


"""For Wgan"""
# train_x_file_path = "./app/assets/images/small_x"
# count = 0
# batch_size = 32
# # train_x = np.array(ImageProcessor.img_to_data_set(train_x_file_path))
# # # (train_x, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
# # train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
# # train_x = (train_x - 127.5) / 127.5
# # data_shape = np.shape(train_x)
# train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
# pdb.set_trace()
# train_x = (train_x - 127.5) / 127.5
# data_shape = np.shape(train_x)
# model = Wgan([data_shape[1], data_shape[2], data_shape[3]], batch_size)

# while True:
    
#     train_ds = (
#         tf.data.Dataset.from_tensor_slices(train_x)
#         .shuffle(buffer_size=batch_size)
#         .batch(batch_size)
#     )
#     if len(train_x) == batch_size:
#         model.train(train_ds, count % 20 == 0)

#     train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
#     train_x = (train_x - 127.5) / 127.5
#     count += 1

"""for Autoencoder"""
# train_x_file_path = "./app/assets/images/small_x"
# batch_size=1
# train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
# train_x = (train_x - 127.5) / 127.5
# data_shape = np.shape(train_x)
# model = Autoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size, 'AUTOENCODER', False)

# count = 0
# while True:
    
#     train_ds = (
#         tf.data.Dataset.from_tensor_slices(train_x)
#         .batch(batch_size)
#     )

#     if len(train_x) == batch_size:
#         model.train(train_ds, None, count % 20 == 0)

#     train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
#     train_x = (train_x - 127.5) / 127.5
#     count += 1
"""for autoencoder generate"""
# train_x_file_path = "./app/assets/images/y/0030.jpg"
# batch_size=1
# train_x = np.array([ImageProcessor.parse_function(train_x_file_path)])
# train_x = (train_x - 127.5) / 127.5
# data_shape = np.shape(train_x)
# model = Autoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size, 'AUTOENCODER', False)
# train_ds = (
#     tf.data.Dataset.from_tensor_slices(train_x)
#     .shuffle(buffer_size=batch_size)
#     .batch(batch_size)
# )
# ds = iter(train_ds)
# images = next(ds)
# encoder = model.encoder()
# decoder = model.decoder()
# output = encoder(images)
# result = decoder(output)
# cv2.imshow('result', np.array(result * 127.5 + 127.5).astype(np.uint8)[0])
# cv2.waitKey(0)

"""For context autoencoder"""
# train_x_file_path = "./app/assets/images/small_x"
# batch_size=16
# train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
# masked_x = ImageProcessor.mask_randomly(train_x, (20,20))

# masked_x = (masked_x - 127.5) / 127.5
# train_x = (train_x - 127.5) / 127.5

# data_shape = np.shape(train_x)
# model = ContextAutoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size)
# count = 0
# while True:
#     train_ds = (
#         tf.data.Dataset.from_tensor_slices(masked_x)
#         .batch(batch_size)
#     )

#     train_ys = (
#         tf.data.Dataset.from_tensor_slices(train_x)
#         .batch(batch_size)
#     )
#     if len(train_x) == batch_size:
#         model.train(train_ds, train_ys, count % 20 == 0)
#     train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))
#     while len(train_x) != batch_size:
#         train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size))

#     masked_x = ImageProcessor.mask_randomly(train_x, (20,20))
#     masked_x = (masked_x - 127.5) / 127.5
#     train_x = (train_x - 127.5) / 127.5
#     count += 1


"""For upscaling train"""
# train_x_file_path = "./app/assets/images/small_x"
# train_y_file_path = "./app/assets/images/2x_small_x"
# count = 1
# xs, ys = ImageProcessor.load_paired_data_set_batch(train_x_file_path, train_y_file_path, batch_size)
# train_x = np.array(xs)
# train_y = np.array(ys)
# train_x = (train_x - 127.5) / 127.5
# train_y = (train_y - 127.5) / 127.5
# data_shape = np.shape(train_x)
# y_shape = np.shape(train_y)
# model = Autoencoder([data_shape[1], data_shape[2], data_shape[3]], batch_size, True, (y_shape[1], y_shape[2], y_shape[3]))

# while True:
    
#     train_xs = (
#         tf.data.Dataset.from_tensor_slices(train_x)
#         .batch(batch_size)
#     )
#     train_ys = (
#         tf.data.Dataset.from_tensor_slices(train_y)
#         .batch(batch_size)
#     )
#     # data_set = encoder.img_to_data_set_batch(train_x_file_path, batch_size)
#     if len(train_x) == batch_size:
#         model.train(train_xs, train_ys, count % 20 == 0, count % 200 == 0)

#     xs, ys = ImageProcessor.load_paired_data_set_batch(train_x_file_path, train_y_file_path, batch_size)
#     train_x = np.array(xs)
#     train_y = np.array(ys)
#     train_x = (train_x - 127.5) / 127.5
#     train_y = (train_y - 127.5) / 127.5
#     count += 1

"""for image inpainting with partial conv"""
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# traingen = ImageMasker(x_train, x_train)
train_x_file_path = "./app/assets/images/2x_small_x"
train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size, (216, 384, 3)))

img_shape = train_x[0].shape
padding_y_n = 32 - img_shape[0] % 32
padding_x_n = 32 - img_shape[1] % 32
padded_imgs = np.array([np.stack([np.pad(img[:,:,c], pad_width=((0,padding_y_n), (0,padding_x_n)), mode='constant', constant_values=255) for c in range(3)], axis=2) for img in train_x])
padded_img_shape = padded_imgs[0].shape
# v1
# model_v1 = InpaintingModel().prepare_model()
# model_v1.compile(optimizer='adam', loss='mean_absolute_error')
# v2
model_v2 = InpaintingModelV2().prepare_model(padded_img_shape)
model_v2.compile(optimizer='adam', loss='mean_absolute_error')

# count = 0
# while True:
#     print(count)
#     if len(padded_imgs) == batch_size:
#         [masked_xs, masks], ys = ImageProcessor.mask_data_v2(padded_imgs, batch_size, 8)
#         with tf.device("/gpu:0"):
#             model.fit(
#                 [masked_xs, masks],
#                 ys,
#                 epochs=1, 
#                 steps_per_epoch=batch_size, 
#                 use_multiprocessing=True,
#                 callbacks=[
#                     ValidationCallbackV2([[masked_xs, masks], ys])
#                 ] if count % 500 == 0 else []
#             )
#         model.save_weights('trained_parameters/image_inpainting_v2/ckpt')
#     train_x = np.array(ImageProcessor.img_to_data_set_batch(train_x_file_path, batch_size, (216, 384, 3)))
#     padded_imgs = np.array([np.stack([np.pad(img[:,:,c], pad_width=((0,padding_y_n), (0,padding_x_n)), mode='constant', constant_values=255) for c in range(3)], axis=2) for img in train_x])

#     count += 1
line_weight = 1
# masked_xs, ys = ImageProcessor.mask_data(padded_imgs, batch_size, line_weight)
# inpainted_image_v1 = model_v1.predict(np.expand_dims(masked_xs[0], axis=0))
[masked_xs_v2, masks], ys = ImageProcessor.mask_data_v2(padded_imgs, batch_size, line_weight)
sample_idx = 0
inpainted_image_v2 = model_v2.predict([masked_xs_v2[sample_idx].reshape((1,)+masked_xs_v2[sample_idx].shape), masks[sample_idx].reshape((1,)+masks[sample_idx].shape)])
# cv2.imshow('input',np.array(masked_xs)[0])
# cv2.imshow('inpainted_image_v1',np.array(inpainted_image_v1)[0])
# cv2.imwrite('inpainted_image_v2.jpg', inpainted_image_v2[0] * 255)
# cv2.imwrite('input_v2.jpg', masked_xs_v2[0] * 255)
cv2.imshow('input_v2',np.array(masked_xs_v2)[0])
cv2.imshow('inpainted_image_v2',np.array(inpainted_image_v2)[0])
cv2.waitKey(0)
