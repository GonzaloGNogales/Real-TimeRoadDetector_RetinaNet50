import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping


class MultiClassClassifierAugmentation:
    def __init__(self, t_path='./dataset/train',
                 v_path='./dataset/validation',
                 i_size=(150, 150)):
        self.train_path = t_path
        self.val_path = v_path
        self.input_size = i_size
        self.train_generator = None
        self.validation_generator = None
        self.model = None

    def set_up_data_generator(self, b_size=32):
        train_data_generator = ImageDataGenerator(rescale=1. / 255.,
                                                  rotation_range=30,
                                                  shear_range=0.2,
                                                  zoom_range=0.3,
                                                  brightness_range=[0.2, 1.0],
                                                  horizontal_flip=True,
                                                  fill_mode='nearest')
        self.train_generator = train_data_generator.flow_from_directory(self.train_path,
                                                                        batch_size=b_size,
                                                                        class_mode='categorical',
                                                                        target_size=self.input_size)

        validation_data_generator = ImageDataGenerator(rescale=1. / 255.)
        self.validation_generator = validation_data_generator.flow_from_directory(self.val_path,
                                                                                  batch_size=b_size,
                                                                                  class_mode='categorical',
                                                                                  target_size=self.input_size)

    def define_model_architecture_v1(self, num_classes=9, opt=Adam(learning_rate=0.001)):
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

    def define_model_architecture_v2(self, num_classes=9, opt=Adam(learning_rate=0.001)):
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

    def train(self, epochs=25, verbose=1, steps_per_epoch=True, architecture_ver='v2'):
        history = None
        if self.model is not None and self.train_generator is not None and self.validation_generator is not None:
            if steps_per_epoch:
                history = self.model.fit(self.train_generator,
                                         epochs=epochs,
                                         steps_per_epoch=100,
                                         verbose=verbose,
                                         validation_data=self.validation_generator,
                                         callbacks=[ModelCheckpoint('./models/multiclass_augmentation_' + str(self.input_size) + '_' + architecture_ver + '_spe_save.h5',
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True,
                                                                    verbose=1),
                                                    EarlyStopping(
                                                        monitor='val_loss',
                                                        mode='min',
                                                        patience=6,
                                                        min_delta=0.005,
                                                        verbose=1)
                                                    ])
            else:
                history = self.model.fit(self.train_generator,
                                         epochs=epochs,
                                         verbose=verbose,
                                         validation_data=self.validation_generator,
                                         callbacks=[ModelCheckpoint('./models/multiclass_augmentation_' + str(self.input_size) + '_' + architecture_ver + '_save.h5',
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True,
                                                                    verbose=1),
                                                    EarlyStopping(
                                                        monitor='val_loss',
                                                        mode='min',
                                                        patience=6,
                                                        min_delta=0.005,
                                                        verbose=1)
                                                    ])
        return history

    def evaluate(self):
        print('Model score:', self.model.evaluate(self.validation_generator))
