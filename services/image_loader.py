import os
import sys
import numpy as np
import cv2
import random
import pdb

class ImageLoader:
    @classmethod
    def img_to_data_set(cls,files_path, img_shape=(108, 192, 3)):
        filelist = os.listdir(files_path)
        filelist.sort()
        images = []
        count = 0
        for file_name in filelist:
            process_rate = float(count/len(filelist)) * 100
            temp_img = ImageLoader.parse_function(files_path + '\\' + file_name)
            if np.shape(temp_img) == img_shape:
                images.append(temp_img) 
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return images
    @classmethod
    def parse_function(cls, filename):
        os.path.join(os.getcwd(), filename)
        path = filename.replace('./', '\\')
        path = path.replace('/', '\\')
        path = os.getcwd() + path
        image_np_array = cv2.imread(path)
        return image_np_array

    @classmethod
    def img_to_data_set_batch(cls, files_path, batch_size):
        # filelist = glob.glob(os.path.join(files_path, '*'))
        filelist = os.listdir(files_path)
        end_index = len(filelist) -1
        index_start = random.randint(0,end_index)
        index_end = index_start + batch_size
        if index_end > end_index:
            index_end = index_start - batch_size
        name_list = filelist[index_start:index_end]
        images = []
        count = 0
        
        for file_name in name_list:
            process_rate = float(count/len(name_list)) * 100
            temp_img = ImageLoader.parse_function(files_path + '\\' + file_name)
            if np.shape(temp_img) == (108, 192, 3):
                images.append(temp_img) 
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return images

    @classmethod
    def load_single_file(cls, file_path):
        filelist = os.listdir(files_path)