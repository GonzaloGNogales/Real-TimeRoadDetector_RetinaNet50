import os
import zipfile
import cv2
import numpy as np
import random
import shutil
from shutil import copyfile


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
        class_folder_correspondence = {0: 'cars', 1: 'motos', 2: 'trucks', 3: 'pedestrians', 4: 'forbid_signals', 5: 'warning_signals', 6: 'stop_signals', 7: 'yield_signals'}
        class_indexes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        labels_list = os.listdir(self.labels_train_raw_path)
        labels_list.remove('classes.txt')
        images = os.listdir(self.train_raw_path)
        images = map(lambda l: int(l[:-4]), images)
        images = list(images)
        images.sort()
        print(labels_list)
        print(images)

        images_map = {}
        for i in images:
            loaded_img = cv2.imread(self.train_raw_path + '/' + str(i) + '.png')
            images_map[str(i) + '.txt'] = loaded_img
            print(str(i) + ' image successfully loaded => {', str(i) + '.txt :', loaded_img.shape, '}')

        print('!!!!!!!!!!!!! Images loaded successfully !!!!!!!!!!!!!')

        for file_name in labels_list:
            labels = open(self.labels_train_raw_path + '/' + file_name, "r")
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
                cv2.imwrite(os.path.join('D:\\DatasetTFG\\train\\' + class_folder_correspondence[c],
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


if __name__ == '__main__':
    dataset_preprocessor = DataPreprocessing()
    # dataset_preprocessor.dataset_set_up()


# zip_data_path = "../dataframe.zip"
# zip_ref = zipfile.ZipFile(zip_data_path, 'r')
# os.mkdir("./data_management_tmp")
# zip_ref.extractall('./data_management_tmp')
# zip_ref.close()
#
# print("Number of car images:", len(os.listdir('./data_management_tmp/RoadImages/Cars/')))
# print("Number of truck images:", len(os.listdir('./data_management_tmp/RoadImages/Trucks/')))
# print("Number of motorbike images:", len(os.listdir('./data_management_tmp/RoadImages/Motorbikes/')))
# print("Number of crosswalk images:", len(os.listdir('./data_management_tmp/RoadImages/Crosswalks/')))
#
# try:
#     os.mkdir("./RoadClassifier")
#     os.mkdir("./RoadClassifier/training")
#     os.mkdir("./RoadClassifier/validation")
#     os.mkdir("./RoadClassifier/training/cars")
#     os.mkdir("./RoadClassifier/training/trucks")
#     os.mkdir("./RoadClassifier/training/motorbikes")
#     os.mkdir("./RoadClassifier/training/crosswalks")
#     os.mkdir("./RoadClassifier/validation/cars")
#     os.mkdir("./RoadClassifier/validation/trucks")
#     os.mkdir("./RoadClassifier/validation/motorbikes")
#     os.mkdir("./RoadClassifier/validation/crosswalks")
# except OSError:
#     pass
