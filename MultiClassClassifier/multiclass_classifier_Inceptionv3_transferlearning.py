import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class MultiClassClassifierInceptionV3TransferLearning:
    def __init__(self, t_path, v_path, i_size, b_size=20):
        self.train_path = t_path
        self.val_path = v_path
        self.input_size = i_size
        self.train_generator = None
        self.validation_generator = None
        self.model = None
        self.local_weights_file = './MultiClassClassifier/inceptionv3_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

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
        train_data_generator = ImageDataGenerator(rescale=1. / 255.,
                                                  rotation_range=15,
                                                  shear_range=0.2,
                                                  zoom_range=0.2,
                                                  brightness_range=[0.35, 1.0],
                                                  horizontal_flip=True,
                                                  fill_mode='nearest')
        self.train_generator = train_data_generator.flow_from_directory(self.train_path,
                                                                        batch_size=self.batch_size,
                                                                        class_mode='categorical',
                                                                        target_size=self.input_size)

        validation_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.validation_generator = validation_data_generator.flow_from_directory(self.val_path,
                                                                                  batch_size=self.batch_size,
                                                                                  class_mode='categorical',
                                                                                  target_size=self.input_size)

    def compile_model(self, num_classes=10, opt=Adam(learning_rate=0.0001)):
        self.input_size += (3,)
        pre_trained_model = InceptionV3(input_shape=self.input_size,
                                        include_top=False,
                                        weights=None)

        pre_trained_model.load_weights(self.local_weights_file)

        for layer in pre_trained_model.layers:
            layer.trainable = False

        last_layer = pre_trained_model.get_layer('mixed7')
        print('last layer output shape: ', last_layer.output_shape)
        last_output = last_layer.output

        x = layers.Flatten()(last_output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)

        self.model = Model(pre_trained_model.input, x)

        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=200, verbose=1):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            history = self.model.fit(self.train_generator,
                                     epochs=epochs,
                                     steps_per_epoch=self.train_length // self.batch_size,
                                     verbose=verbose,
                                     validation_data=self.validation_generator,
                                     validation_steps=self.val_length // self.batch_size,
                                     callbacks=[
                                         ModelCheckpoint('./results/non_realtime_results/models/multiclass_transferlearning_inceptionv3_save.h5',
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         verbose=1),
                                         EarlyStopping(
                                             monitor='val_loss',
                                             mode='min',
                                             patience=10,
                                             min_delta=0.0005,
                                             verbose=1)
                                         ])
        return history

    def evaluate(self):
        return self.model.evaluate(self.validation_generator)

    def load_model(self):
        self.model = tf.keras.models.load_model('./results/non_realtime_results/models/multiclass_transferlearning_inceptionv3_save.h5')

    def predict(self, path):
        return self.model.predict(path)
