import os
import sys
import numpy as np
import cv2
import random
import pdb

class ImageProcessor:
    @classmethod
    def img_to_data_set(cls,files_path, img_shape=(108, 192, 3)):
        filelist = os.listdir(files_path)
        filelist.sort()
        images = []
        count = 0
        for file_name in filelist:
            process_rate = float(count/len(filelist)) * 100
            temp_img = cls.parse_function(files_path + '\\' + file_name)
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
    def img_to_data_set_batch(cls, files_path, batch_size, img_shape):
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
            temp_img = cls.parse_function(files_path + '\\' + file_name)
            if np.shape(temp_img) == img_shape:
                images.append(temp_img) 
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return images

    @classmethod
    def load_paired_data_set_batch(cls, x_file_path, y_file_path, batch_size):
        x_filelist = os.listdir(x_file_path)
        y_filelist = os.listdir(y_file_path)
        end_index = len(x_filelist) -1
        index_start = random.randint(0,end_index)
        index_end = index_start + batch_size
        if index_end > end_index:
            index_end = index_start - batch_size
        name_list = x_filelist[index_start:index_end]
        xs = []
        ys = []
        count = 0
        
        for file_name in name_list:
            process_rate = float(count/len(name_list)) * 100
            temp_x_img = cls.parse_function(x_file_path + '\\' + file_name)
            temp_y_img = cls.parse_function(y_file_path + '\\' + file_name)
            if np.shape(temp_x_img) == (108, 192, 3):
                xs.append(temp_x_img) 
                ys.append(temp_y_img)
            
            sys.stdout.write("\rDoing thing %i ％" % process_rate)
            sys.stdout.flush()
            count += 1
        # dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return [xs, ys]

    @classmethod
    def mask_randomly(cls, imgs, mask_shape):
        # pdb.set_trace()
        y1 = np.random.randint(0, imgs.shape[1] - mask_shape[1], imgs.shape[0])
        y2 = y1 + mask_shape[1]
        x1 = np.random.randint(0, imgs.shape[2] - mask_shape[0], imgs.shape[0])
        x2 = x1 + mask_shape[0]
        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty((imgs.shape[0], mask_shape[1], mask_shape[0], imgs.shape[3]))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            # missing_parts[i] = masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs

    @classmethod
    def mask_data(cls, imgs, batch_size, line_weight=3):
        # X_batch is a matrix of masked images used as input
        img_shape = imgs[0].shape
        X_batch = np.empty((batch_size, img_shape[0], img_shape[1], img_shape[2])) # Masked image
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((batch_size, img_shape[0], img_shape[1], img_shape[2])) # Original image

        ## Iterate through random indexes
        for i in range(batch_size):
            image_copy = imgs[i].copy()
    
            ## Get mask associated to that image
            masked_image = cls.createMask(image_copy, line_weight)
            
            X_batch[i,] = masked_image/255
            y_batch[i,] = image_copy/255
        
        return X_batch, y_batch

    @classmethod
    def createMask(cls, img, line_weight=3):
        ## Prepare masking matrix
        img_shape = img.shape
        mask = np.full((img_shape[0],img_shape[1],img_shape[2]), 255, np.uint8)
        for _ in range(10):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, img_shape[0]), np.random.randint(1, img_shape[0])
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, img_shape[1]), np.random.randint(1, img_shape[1])
            # Get random thickness of the line drawn
            # thickness = np.random.randint(1, line_weight)
            # Draw black line on the white mask
            cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),line_weight)

        # Perforn bitwise and operation to mak the image
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    @classmethod
    def mask_data_v2(cls,imgs, batch_size, paths, line_weight=3):
        img_shape = imgs[0].shape
        # Masked_images is a matrix of masked images used as input
        masked_x_batch = np.empty((batch_size, img_shape[0], img_shape[1], img_shape[2])) # Masked image
        # Mask_batch is a matrix of binary masks used as input
        mask_batch = np.empty((batch_size, img_shape[0], img_shape[1], img_shape[2])) # Binary Masks
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((batch_size, img_shape[0], img_shape[1], img_shape[2])) # Original image
        for i in range(batch_size):
            image_copy = imgs[i].copy()
        
            ## Get mask associated to that image
            masked_image, mask = cls.create_mask_v2(image_copy, paths[i], line_weight)
            
            masked_x_batch[i,] = masked_image/255
            mask_batch[i,] = mask/255
            y_batch[i,] = image_copy/255
        
        ## Return mask as well because partial convolution require the same.
        return [masked_x_batch, mask_batch], y_batch

    @classmethod    
    def create_mask_v2(cls, img, paths, line_weight=3):
        img_shape = img.shape
        mask = np.full(img_shape, 255, np.uint8) ## White background
        x1, y1 = paths[0]
        for x2, y2 in paths:
            # # Get random x locations to start line
            # x1, x2 = np.random.randint(1, img_shape[0]), np.random.randint(1, img_shape[0])
            # # Get random y locations to start line
            # y1, y2 = np.random.randint(1, img_shape[1]), np.random.randint(1, img_shape[1])
            # # Get random thickness of the line drawn
            # # thickness = np.random.randint(1, line_weight)
            # Draw black line on the white mask
            cv2.line(mask,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),line_weight)
            x1 = x2
            y1 = y2
            
        ## Mask the image
        masked_image = img.copy()
        masked_image[mask==0] = 255
        
        return masked_image, mask

class ImageMasker:
    'Generates data for Keras'
    def __init__(self, X, y, batch_size=32, dim=(32, 32), n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size 
        self.X = X 
        self.y = y
        img_shape = dim
        img_shape[2] = n_channels
        self.shuffle = shuffle
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs):
        # X_batch is a matrix of masked images used as input
        X_batch = np.empty((self.batch_size, img_shape[0], img_shape[1], img_shape[2])) # Masked image
        # y_batch is a matrix of original images used for computing error from reconstructed image
        y_batch = np.empty((self.batch_size, img_shape[0], img_shape[1], img_shape[2])) # Original image

        ## Iterate through random indexes
        for i, idx in enumerate(idxs):
            image_copy = self.X[idx].copy()
    
            ## Get mask associated to that image
            masked_image = self.__createMask(image_copy)
            
            X_batch[i,] = masked_image/255
            y_batch[i] = self.y[idx]/255
        
        return X_batch, y_batch

    def __createMask(self, img):
        ## Prepare masking matrix
        mask = np.full((32,32,3), 255, np.uint8)
        for _ in range(np.random.randint(1, 10)):
            # Get random x locations to start line
            x1, x2 = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random y locations to start line
            y1, y2 = np.random.randint(1, 32), np.random.randint(1, 32)
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, 3)
            # Draw black line on the white mask
            cv2.line(mask,(x1,y1),(x2,y2),(1,1,1),thickness)

        # Perforn bitwise and operation to mak the image
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image