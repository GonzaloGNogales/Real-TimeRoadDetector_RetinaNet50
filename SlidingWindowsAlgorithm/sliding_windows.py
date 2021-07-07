import shutil
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
from SlidingWindowsAlgorithm.Window import *
from SlidingWindowsAlgorithm.Detection import *
import numpy as np
import os
import torch
import torchvision.ops.boxes as bops
from DeepLearningUtilities.progress_bar import *


def sliding_windows_test(model_name, model, input_size=(150, 150), show=False):
    file = open(model_name + '.txt', 'a')

    for c in os.listdir('./dataset/validation'):
        for im in os.listdir('./dataset/validation/' + c):
            img_path = './dataset/validation/' + c + '/' + im
            img = image.load_img(img_path, target_size=input_size)
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255

            if show:
                plt.imshow(img_tensor[0])
                plt.axis('off')
                plt.show()

            prediction = model.predict(img_tensor)
            prediction_idx = np.argmax(prediction[0])
            file.write('Expected: [' + c[:-1] + '] -> ')
            if prediction_idx == 0:
                file.write('Prediction: [car]\n')
            elif prediction_idx == 1:
                file.write('Prediction: [forbid_signal]\n')
            elif prediction_idx == 2:
                file.write('Prediction: [moto]\n')
            if prediction_idx == 3:
                file.write('Prediction: [none]\n')
            elif prediction_idx == 4:
                file.write('Prediction: [pedestrian]\n')
            elif prediction_idx == 5:
                file.write('Prediction: [stop_signal]\n')
            elif prediction_idx == 6:
                file.write('Prediction: [truck]\n')
            elif prediction_idx == 7:
                file.write('Prediction: [warning_signal]\n')
            elif prediction_idx == 8:
                file.write('Prediction: [yield_signal]\n')

    file.close()


def sliding_windows(model_name, model, input_size, source_path):
    classes = {0: 'car', 1: 'forbid', 2: 'moto', 3: 'none', 4: 'ped', 5: 'stop', 6: 'truck',
               7: 'warning', 8: 'yield'}
    class_colors = {0: (255, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0),
                    3: (0, 0, 0), 4: (255, 0, 255), 5: (255, 255, 255),
                    6: (230, 40, 20), 7: (0, 200, 255), 8: (0, 255, 115)}
    # list of Windows
    windows = [Window(0, 0, 100, 100),  #
               Window(0, 0, 140, 140),  #
               Window(0, 0, 270, 240),  #
               Window(0, 0, 240, 550),  #
               Window(0, 0, 620, 620),  #
               Window(0, 0, 725, 325),  #
               Window(0, 0, 400, 200)]  #

    # Prepare the detection results directory
    # Check if results folder already exists and clear it, if not create it
    result_dir = './sliding_windows_result/' + model_name + '/'
    if os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    detections_file = open(result_dir + model_name + '_detections.txt', 'a')
    detections_file.truncate(0)

    it = 0
    total = len(os.listdir(source_path))
    if total != 0:
        progress_bar(it, total, prefix='Performing sliding windows detection on frame ' + str(it) + ': ', suffix='Complete', length=50)
    for im in os.listdir(source_path):
        it += 1
        progress_bar(it, total, prefix='Performing sliding windows detection on frame ' + str(it) + ': ', suffix='Complete', length=50)
        car_detections = list()
        forbid_signal_detections = list()
        moto_detections = list()
        pedestrian_detections = list()
        stop_signal_detections = list()
        truck_detections = list()
        warning_signal_detections = list()
        yield_signal_detections = list()
        detections = {0: car_detections, 1: forbid_signal_detections, 2: moto_detections,
                      4: pedestrian_detections, 5: stop_signal_detections, 6: truck_detections,
                      7: warning_signal_detections, 8: yield_signal_detections}

        img_path = source_path + '/' + im
        final_img = cv2.imread(img_path)
        img = image.load_img(img_path)
        img_tensor = image.img_to_array(img)
        h, w, _ = img_tensor.shape
        img_tensor /= 255
        for window in windows:
            for j in range(window.y2, h, 150):  # Step of 50 from the starting y2
                for i in range(window.x2, w, 100):  # Step of 50 from the starting x2
                    x1 = i - window.x2
                    y1 = j - window.y2
                    x2 = i
                    y2 = j
                    # Extract the region from the original image to predict on the actual window
                    crop_window = img_tensor[y1:y2, x1:x2]
                    crop_window = cv2.resize(crop_window, input_size)
                    crop_window = np.expand_dims(crop_window, axis=0)
                    prediction = model.predict(crop_window)
                    prediction = prediction[0]
                    prediction_idx = int(np.argmax(prediction))
                    # Filter all the predictions with lower than 60% probability
                    if classes[prediction_idx] != 'none' and prediction[prediction_idx] >= 0.8:
                        # Save the detection on it corresponding detection set
                        detections[prediction_idx].append((Detection(x1, y1, x2, y2, prediction[prediction_idx],
                                                                     prediction_idx),
                                                           prediction[prediction_idx]))

        # Perform Non-Maximum Suppression Algorithm on each set of detections for each class
        final_detections = list()
        for l_det in detections.values():
            if l_det:
                l_det = sorted(l_det, key=lambda x: x[1], reverse=True)
                l_det = [x[0] for x in l_det]
                nms(l_det, final_detections)

        # Draw bounding boxes and label with prediction probability
        for d in final_detections:
            x1, y1, x2, y2, p, c_idx = d.unpack()
            cv2.rectangle(final_img, (x1, y1), (x2, y2), class_colors[c_idx], 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = classes[c_idx] + ' ' + str('{0:.2g}'.format(p))
            if y1 - 30 > 50:
                cv2.rectangle(final_img, (x1, y1 - 30), (x2+20, y1), class_colors[c_idx], -1)
                cv2.putText(final_img, text, (x1, y1 - 10), font, 0.6, (0, 0, 0), 2)
            else:
                cv2.rectangle(final_img, (x1, y2 + 30), (x2+20, y2), class_colors[c_idx], -1)
                cv2.putText(final_img, text, (x1, y2 + 15), font, 0.6, (0, 0, 0), 2)
            detections_file.write(str(im) + ';'
                                  + str(x1) + ';'
                                  + str(y1) + ';'
                                  + str(x2) + ';'
                                  + str(y2) + ';'
                                  + str(classes[c_idx]) + ';'
                                  + str('{0:.4g}'.format(p))
                                  + '\n')
        # Save the final detected image to the directory
        cv2.imwrite(result_dir + im, final_img)
    detections_file.close()

    return result_dir


# Non-maximum Suppression Algorithm
def nms(det, final_det):
    to_remove = list()
    while det:
        to_remove.clear()
        selected_detection = det.pop()
        final_det.append(selected_detection)
        box1 = torch.tensor(selected_detection.get_box_tensor(), dtype=torch.float)
        for d in det:
            box2 = torch.tensor(d.get_box_tensor(), dtype=torch.float)
            iou = float(bops.box_iou(box1, box2)[0][0])
            if iou >= 0.1:
                to_remove.append(d)
        for r in to_remove:
            det.remove(r)


def clean_test_video():
    test_dir = '../test_video/'
    s = set(os.listdir(test_dir))
    res_dir = '../sliding_windows_result/ResNet50_TL/'
    to_remove = list()
    for f in os.listdir(res_dir):
        if f in s:
            to_remove.append(f)
    for f in to_remove:
        os.remove(test_dir+f)
