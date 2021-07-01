import argparse
from MultiClassClassifier.multiclass_classifier_no_augmentation import *
from MultiClassClassifier.multiclass_classifier_augmentation import *
from MultiClassClassifier.multiclass_classifier_VGG16based_model import *
from DeepLearningUtilities.metrics_analyzer import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a convolutional neural network')
    parser.add_argument(
        '--train_path', type=str, default='./training/', help='Path to training dir')
    parser.add_argument(
        '--val_path', type=str, default='./validation/', help='Path to validation dir')
    parser.add_argument(
        '--dnn', type=str, default='MCNA', help='String referring to the neural network selection')

    args = parser.parse_args()
    dnn = None
    model_ver = None
    spe = True

    # Instantiate a custom DNN to train
    if args.dnn == 'MCNA':
        dnn = MultiClassClassifierNoAugmentation()  # Instantiate HERE
        model_ver = 'v1'
    elif args.dnn == 'MCA':
        dnn = MultiClassClassifierAugmentation()
        model_ver = 'v1'
    elif args.dnn == 'MCA-v1-150x150':
        dnn = MultiClassClassifierAugmentation(i_size=(150, 150))
        model_ver = 'v1'
        spe = False
    elif args.dnn == 'MCA-v1-150x150-spe':
        dnn = MultiClassClassifierAugmentation(i_size=(150, 150))
        model_ver = 'v1'
        spe = True
    elif args.dnn == 'MCA-v1-224x224':
        dnn = MultiClassClassifierAugmentation(i_size=(224, 224))
        model_ver = 'v1'
        spe = False
    elif args.dnn == 'MCA-v1-224x224-spe':
        dnn = MultiClassClassifierAugmentation(i_size=(224, 224))
        model_ver = 'v1'
        spe = True
    elif args.dnn == 'MCA-v2-150x150':
        dnn = MultiClassClassifierAugmentation(i_size=(150, 150))
        model_ver = 'v2'
        spe = False
    elif args.dnn == 'MCA-v2-150x150-spe':
        dnn = MultiClassClassifierAugmentation(i_size=(150, 150))
        model_ver = 'v2'
        spe = True
    elif args.dnn == 'MCA-v2-224x224':
        dnn = MultiClassClassifierAugmentation(i_size=(224, 224))
        model_ver = 'v2'
        spe = False
    elif args.dnn == 'MCA-v2-224x224-spe':
        dnn = MultiClassClassifierAugmentation(i_size=(224, 224))
        model_ver = 'v2'
        spe = True
    elif args.dnn == 'VGG16':
        dnn = MultiClassClassifierVGG16(num_classes=9)
    else:
        raise ValueError('Wrong DNN type :(')

    if dnn is not None:
        # Set up data generator for pre processing the images and fixing batch size
        dnn.set_up_data_generator()

        # Model architecture definition and compilation
        if model_ver == 'v1':
            dnn.define_model_architecture_v1()
        elif model_ver == 'v2':
            dnn.define_model_architecture_v2()
        else:
            dnn.compile_model()

        # Train the DNN
        if model_ver is not None:
            h = dnn.train(epochs=50, steps_per_epoch=spe, architecture_ver=model_ver)
        else:
            h = dnn.train(epochs=500)

        # Evaluate the model to get the loss value and the metrics values of the model in validation
        loss, accuracy = dnn.evaluate()

        # Plot metrics from training when finished
        plot_metrics_legend(h, args.dnn, loss, accuracy)
