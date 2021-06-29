from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model


class MultiClassClassifierNoAugmentation:
    def __init__(self, t_path='D:\\DatasetTFG\\dataset\\train',
                 v_path='D:\\DatasetTFG\\dataset\\validation',
                 i_size=(150, 150)):
        self.train_path = t_path
        self.val_path = v_path
        self.input_size = i_size
        self.train_generator = None
        self.validation_generator = None
        self.model = None

    def set_up_data_generator(self, b_size=10):
        train_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.train_generator = train_data_generator.flow_from_directory(self.train_path,
                                                                        batch_size=b_size,
                                                                        class_mode='categorical',
                                                                        target_size=self.input_size)

        validation_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.validation_generator = validation_data_generator.flow_from_directory(self.val_path,
                                                                                  batch_size=b_size,
                                                                                  class_mode='categorical',
                                                                                  target_size=self.input_size)

    def define_model_architecture(self, num_classes=9, opt=Adam(learning_rate=0.001)):
        self.input_size += (3,)
        print('Model input size:', self.input_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=self.input_size),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=100, verbose=1):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            history = self.model.fit(self.train_generator,
                                     epochs=epochs,
                                     verbose=verbose,
                                     validation_data=self.validation_generator,
                                     callbacks=[ModelCheckpoint('./models/multiclass_no_augmentation_save.h5',
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True,
                                                                verbose=1),
                                                EarlyStopping(
                                                    monitor='val_loss',
                                                    mode='min',
                                                    patience=5,
                                                    min_delta=0.05,
                                                    verbose=1)
                                                ])
        return history

    def predict(self):
        load_model('./models/multiclass_no_augmentation_save.h5')
        return
