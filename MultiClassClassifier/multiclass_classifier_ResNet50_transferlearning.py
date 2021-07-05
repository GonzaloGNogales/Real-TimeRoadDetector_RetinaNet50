import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class MultiClassClassifierResNet50TransferLearning:
    def __init__(self, t_path='./dataset/train',
                 v_path='./dataset/validation',
                 i_size=(224, 224),
                 b_size=20):
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

    def set_up_data_generator(self):
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

    def compile_model(self, num_classes=9, opt='SGD'):
        self.input_size += (3,)
        inputs = tf.keras.layers.Input(self.input_size)
        feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                  include_top=False,
                                                                  weights='imagenet')(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        final_classification = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification")(x)

        self.model = Model(inputs=inputs, outputs=final_classification)

        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=100, verbose=1):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            history = self.model.fit(self.train_generator,
                                     epochs=epochs,
                                     steps_per_epoch=self.train_length // self.batch_size,
                                     verbose=verbose,
                                     validation_data=self.validation_generator,
                                     validation_steps=self.val_length // self.batch_size,
                                     callbacks=[
                                         ModelCheckpoint('./models_TL/multiclass_transferlearning_resnet50_saveweights.h5',
                                                         monitor='val_loss',
                                                         mode='min',
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         verbose=1),
                                         EarlyStopping(
                                             monitor='val_loss',
                                             mode='min',
                                             patience=10,
                                             min_delta=0.0005,
                                             verbose=1)
                                         ])

        model_json_architecture = self.model.to_json()
        with open('./models_TL/multiclass_transferlearning_resnet50.json', 'w') as json_file:
            json_file.write(model_json_architecture)

        return history

    def evaluate(self):
        return self.model.evaluate(self.validation_generator)

    def load_model_111(self):
        self.model = tf.keras.models.load_model('./models_TL/multiclass_transferlearning_resnet50_save.h5')

    def load_model(self):
        with open('./models_TL/multiclass_transferlearning_resnet50.json', 'r') as json_file:
            json_loaded_model = json_file.read()
        self.model = tf.keras.models.model_from_json(json_loaded_model)
        self.model.load_weights('./models_TL/multiclass_transferlearning_resnet50_saveweights.h5')

    def predict(self, path):
        return self.model.predict(path)
