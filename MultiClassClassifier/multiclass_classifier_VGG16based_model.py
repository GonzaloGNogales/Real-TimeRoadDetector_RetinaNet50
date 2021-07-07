import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class VGGBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(VGGBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        for i in range(repetitions):
            vars(self)[f'conv2D_{i}'] = tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', padding='same')

        self.max_pool = tf.keras.layers.MaxPooling2D((pool_size, pool_size), (strides, strides))

    def call(self, inputs):
        conv2D_0 = self.conv2D_0
        x = conv2D_0(inputs)

        for i in range(1, self.repetitions):
            conv2D_i = vars(self)[f'conv2D_{i}']
            x = conv2D_i(x)

        max_pool = self.max_pool(x)
        return max_pool


class CustomVGG(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomVGG, self).__init__()
        self.block_a = VGGBlock(64, 3, 2)
        self.block_b = VGGBlock(128, 3, 2)
        self.block_c = VGGBlock(256, 3, 3)
        self.block_d = VGGBlock(512, 3, 3)
        self.block_e = VGGBlock(512, 3, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(256, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class MultiClassClassifierVGG16:
    def __init__(self, t_path='./dataset/train',
                 v_path='./dataset/validation',
                 i_size=(224, 224),
                 b_size=32,
                 num_classes=9):
        self.train_path = t_path
        self.val_path = v_path
        self.input_size = i_size
        self.train_generator = None
        self.validation_generator = None
        self.model = CustomVGG(num_classes)

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
                                                  rotation_range=30,
                                                  shear_range=0.2,
                                                  zoom_range=0.3,
                                                  brightness_range=[0.2, 1.0],
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

    def compile_model(self, opt=Adam(learning_rate=0.0001)):
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, epochs=500, verbose=1):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            history = self.model.fit(self.train_generator,
                                     epochs=epochs,
                                     steps_per_epoch=self.train_length // self.batch_size,
                                     verbose=verbose,
                                     validation_data=self.validation_generator,
                                     validation_steps=self.val_length // self.batch_size,
                                     callbacks=[ModelCheckpoint('./models2/multiclass_vgg16_save.h5',
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
        return history

    def evaluate(self):
        return self.model.evaluate(self.validation_generator)

    def load_model_weights(self):
        self.compile_model()
        self.model.evaluate(self.validation_generator)
        self.model.load_weights('./first_results/first_models/models2/multiclass_vgg16_save.h5')

    def predict(self, path):
        return self.model.predict(path)