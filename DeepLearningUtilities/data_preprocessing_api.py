import os
import zipfile
import cv2
import numpy as np
import random
import shutil
from shutil import copyfile
from PIL import Image


class DataPreprocessing:
    def __init__(self):
        self.window_size = (1080 / 4, 1920 / 7)
        self.train_path_sys = 'C:\\Users\\gonza\\OneDrive\\Escritorio\\BACKUP TFG\\train'
        self.train_path_drive = 'D:\\DatasetTFG\\train'
        self.train_raw_path = 'D:\\DatasetTFG\\train_raw'
        self.labels_train_raw_path = 'D:\\DatasetTFG\\labels_train_raw'
        self.none_class_path = 'D:\\DatasetTFG\\train\\none2'
        self.dataset_origin = 'D:\DatasetTFG\dataset2'
        self.dataset_dest = 'D:\DatasetTFG\dataset'
        self.training = 'D:\\DatasetTFG\\dataset\\train\\none'
        self.validation = 'D:\\DatasetTFG\\dataset\\validation\\none'
        self.mandatory_raw = 'D:\\DatasetTFG\\Labeling process\\Mandatory labeling\\train_raw'
        self.mandatory_labels = 'D:\\DatasetTFG\\Labeling process\\Mandatory labeling\\labels_train_raw'

    def dataset_preparation(self):
        if not os.path.isdir('D:\\DatasetTFG\\cropped_train'):
            os.mkdir('D:\\DatasetTFG\\cropped_train')

        for img in os.listdir(self.train_raw_path):
            image = cv2.imread(self.train_raw_path + '/' + img)  # Read the color image
            print(img[:-4])
            for j in range(0, 4):
                for i in range(0, 7):
                    crop_img = image[int(self.window_size[0] * j):int(self.window_size[0] * j + self.window_size[0]),
                               int(self.window_size[1] * i):int(self.window_size[1] * i + self.window_size[1])]
                    cv2.imwrite(os.path.join('D:\\DatasetTFG\\cropped_train',
                                             'crop_' + img[:-4] + '_' + str(j) + '_' + str(i) + '.png'), crop_img)

    def dataset_refactorer(self):
        if not os.path.isdir('D:\\DatasetTFG\\train\\none'):
            os.mkdir('D:\\DatasetTFG\\train\\none')
        name_idx = 0
        for im in os.listdir(self.none_class_path):
            img = cv2.imread(self.none_class_path + '/' + im)
            print(im[:-4], 'was correctly saved into none directory')
            cv2.imwrite(os.path.join('D:\\DatasetTFG\\train\\none', str(name_idx) + '.png'), img)
            name_idx += 1

    def dataset_refactorer_2(self):
        class_indexes = {'cars': 0, 'motos': 0, 'trucks': 0, 'pedestrians': 0, 'forbid_signals': 0, 'warning_signals': 0, 'stop_signals': 0, 'yield_signals': 0}
        for c in os.listdir(self.train_path_sys):
            if c != 'none':
                for im in os.listdir(self.train_path_sys + '/' + c):
                    img = cv2.imread(self.train_path_sys + '/' + c + '/' + im)
                    cv2.imwrite(os.path.join(self.train_path_drive + '/' + c, str(class_indexes[c]) + '.png'), img)
                    print(im[:-4], 'was correctly saved into', c, 'directory')
                    class_indexes[c] += 1

    def labeled_images_auto_classification(self):
        # format: <class_name> <x> <y> <w> <h>
        class_folder_correspondence = {0: 'cars', 1: 'motos', 2: 'trucks', 3: 'pedestrians', 4: 'forbid_signals', 5: 'warning_signals', 6: 'stop_signals', 7: 'yield_signals', 8: 'mandatory_signals'}
        class_indexes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        labels_list = os.listdir(self.mandatory_labels)
        labels_list.remove('classes.txt')
        images = os.listdir(self.mandatory_raw)
        images = map(lambda l: int(l[:-4]), images)
        images = list(images)
        images.sort()
        print(labels_list)
        print(images)

        images_map = {}
        for i in images:
            loaded_img = cv2.imread(self.mandatory_raw + '/' + str(i) + '.png')
            images_map[str(i) + '.txt'] = loaded_img
            print(str(i) + ' image successfully loaded => {', str(i) + '.txt :', loaded_img.shape, '}')

        print('!!!!!!!!!!!!! Images loaded successfully !!!!!!!!!!!!!')

        for file_name in labels_list:
            labels = open(self.mandatory_labels + '/' + file_name, "r")
            img = images_map[file_name]
            h_img, w_img, _ = img.shape
            print('Image', file_name, '----', 'Image Height:', h_img, '-', 'Image Width:', w_img)

            for label in labels:
                c, x, y, w, h = map(float, label.split())
                c = int(c)
                x1 = int((x-w/2)*w_img)
                x2 = int((x+w/2)*w_img)
                y1 = int((y-h/2)*h_img)
                y2 = int((y+h/2)*h_img)
                print('Label =>', c, int((x-w/2)*w_img), int((x+w/2)*w_img), int((y-h/2)*h_img), int((y+h/2)*h_img), ':', img.shape)
                crop_img = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join('D:\\DatasetTFG\\Labeling process\\Dataset result\\' + class_folder_correspondence[c],
                                         class_folder_correspondence[c] + str(class_indexes[c]) + '.png'), crop_img)
                class_indexes[c] += 1

    def dataset_set_up(self):
        class_indexes = {'cars': 0, 'motos': 0, 'trucks': 0, 'pedestrians': 0, 'forbid_signals': 0,
                         'warning_signals': 0, 'stop_signals': 0, 'yield_signals': 0, 'nones': 0}
        for c in os.listdir(self.dataset_origin):
            if c == 'nones':
                actual_images = os.listdir(self.dataset_origin + '/' + c)
                data_size = len(os.listdir(self.dataset_origin + '/' + c))
                train_size = int(data_size * 0.7)
                for i in range(data_size):
                    if i <= train_size:
                        img_train = cv2.imread(self.dataset_origin + '/' + c + '/' + actual_images[i])
                        cv2.imwrite(os.path.join(self.dataset_dest + '/train/' + c[:-1], c[:-1] + '-' + str(class_indexes[c]) + '.png'), img_train)
                    else:
                        img_val = cv2.imread(self.dataset_origin + '/' + c + '/' + actual_images[i])
                        cv2.imwrite(os.path.join(self.dataset_dest + '/validation/' + c[:-1], c[:-1] + '-' + str(class_indexes[c]) + '.png'), img_val)
                    print(actual_images[i][:-4], 'was correctly saved into', c)
                    class_indexes[c] += 1

    def reformat_none_class(self):
        # 6000 none to training and 2000 none to validation
        target_train = 6000
        target_validation = 2000

        # Build up a new none dataset batch for training from taking random images from source none
        for t in range(target_train):
            random_file = random.choice(os.listdir(self.training))
            print(random_file)
            img = cv2.imread(self.training + '/' + random_file)
            cv2.imwrite(os.path.join('D:\\DatasetTFG\\nonet' + '/' + random_file), img)

        for v in range(target_validation):
            random_file = random.choice(os.listdir(self.validation))
            print(random_file)
            img = cv2.imread(self.validation + '/' + random_file)
            cv2.imwrite(os.path.join('D:\\DatasetTFG\\nonev' + '/' + random_file), img)

    def png_to_jpg(self):
        real_time_ds = '../dataset_realtime/train2'
        for i in os.listdir(real_time_ds):
            im = Image.open(real_time_ds + '/' + i)
            rgb_im = im.convert('RGB')
            rgb_im.save('../dataset_realtime/train/' + i[:-4] + '.jpg')

if __name__ == '__main__':
    dataset_preprocessor = DataPreprocessing()
    dataset_preprocessor.labeled_images_auto_classification()
