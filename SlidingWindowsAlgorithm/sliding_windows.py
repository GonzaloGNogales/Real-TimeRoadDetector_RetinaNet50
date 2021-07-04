from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def sliding_windows(model, input_size=(150, 150)):
    for c in os.listdir('./dataset/validation'):
        none_counter = 0
        for im in os.listdir('./dataset/validation/' + c):
            img_path = './dataset/validation/' + c + '/' + im
            img = image.load_img(img_path, target_size=input_size)
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            prediction = model.predict(img_preprocessed)
            prediction = prediction[0]
            # print('Expected: [' + c[:-1] + '] ->')
            if prediction[0]:
                print('Prediction: [car]')
            elif prediction[1]:
                print('Prediction: [forbid_signal]')
            elif prediction[2]:
                print('Prediction: [moto]')
            if prediction[3]:
                print('Prediction: [none]')
                # none_counter += 1
            elif prediction[4]:
                print('Prediction: [pedestrian]')
            elif prediction[5]:
                print('Prediction: [stop_signal]')
            elif prediction[6]:
                print('Prediction: [truck]')
            elif prediction[7]:
                print('Prediction: [warning_signal]')
            elif prediction[8]:
                print('Prediction: [yield_signal]')

        # correct = len(os.listdir('./dataset/validation/' + c)) - none_counter
        # print('Number of ' + str(c) + ' classified correctly: [' + str(correct) + ']')