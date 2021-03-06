import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class MultiClassClassifierNoAugmentation:
    def __init__(self, t_path, v_path, i_size, b_size=20):
        self.train_path = t_path
        self.val_path = v_path
        self.input_size = i_size
        self.train_generator = None
        self.validation_generator = None
        self.model = None

        t_len = 0
        for d in os.listdir(t_path):
            t_len += len(os.listdir(t_path + '/' + d))
        self.train_length = t_len
        v_len = 0
        for v in os.listdir(v_path):
            v_len += len(os.listdir(v_path + '/' + v))
        self.val_length = v_len
        self.batch_size = b_size

    def set_up_data(self):
        train_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.train_generator = train_data_generator.flow_from_directory(self.train_path,
                                                                        batch_size=self.batch_size,
                                                                        class_mode='categorical',
                                                                        target_size=self.input_size)

        validation_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.validation_generator = validation_data_generator.flow_from_directory(self.val_path,
                                                                                  batch_size=self.batch_size,
                                                                                  class_mode='categorical',
                                                                                  target_size=self.input_size)

    def define_model_architecture_v1(self, num_classes=10, opt=Adam(learning_rate=0.0001)):
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

    def define_model_architecture_v2(self, num_classes=10, opt=Adam(learning_rate=0.0001)):
        self.input_size += (3,)
        print('Model input size:', self.input_size)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_size),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=100, verbose=1, architecture_ver='v1'):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            history = self.model.fit(self.train_generator,
                                     epochs=epochs,
                                     steps_per_epoch=self.train_length // self.batch_size,
                                     verbose=verbose,
                                     validation_data=self.validation_generator,
                                     validation_steps=self.val_length // self.batch_size,
                                     callbacks=[ModelCheckpoint('./results/non_realtime_results/models/multiclass_no_augmentation_' + str(
                                                                self.input_size) + '_' + architecture_ver + '_save.h5',
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True,
                                                                verbose=1),
                                                EarlyStopping(
                                                    monitor='val_loss',
                                                    mode='min',
                                                    patience=40,
                                                    min_delta=0.005,
                                                    verbose=1)
                                                ])
        return history

    def evaluate(self):
        return self.model.evaluate(self.validation_generator)

    def load_model(self, av='v1', mod_input_size=True):
        if mod_input_size:
            self.input_size += (3,)
        self.model = tf.keras.models.load_model('./results/non_realtime_results/models/multiclass_no_augmentation_' + str(
                                                self.input_size) + '_' + av + '_save.h5')

    def predict(self, image):
        return self.model.predict(image)
