import os
import sys
import cv2
import numpy as np
import pdb
sys.path.append(os.path.join(os.getcwd(), 'services/'))
from image_auto_encoder.auto_encoder import AutoEncoder
batch_size = 12
encoder = AutoEncoder([batch_size, 1080, 1920, 3], [3,3,3,3])
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

train_x_file_path = "./app/assets/images/x2"
while True:
    data_set = encoder.img_to_data_set_batch(train_x_file_path, batch_size)
    if len(data_set) != batch_size:
        continue
    encoder.train(data_set)

# batch_size = 1
# encoder = AutoEncoder([batch_size, 1080, 1920, 3], [3,3,3,3])
# train_x_file_path = "./app/assets/images/y"
# data_set = encoder.img_to_data_set(train_x_file_path)
# encoder.generate(data_set)