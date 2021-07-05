from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os


def sliding_windows(model_name, model, input_size=(150, 150), show=False):
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
