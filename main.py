import argparse
from MultiClassClassifier.multiclass_classifier_no_augmentation import *
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

    # Instantiate a custom DNN to train
    if args.dnn == 'MCNA':
        dnn = MultiClassClassifierNoAugmentation()  # Instantiate HERE
    else:
        raise ValueError('Wrong DNN type :(')

    if dnn is not None:
        # Set up data generator for pre processing the images and fixing batch size
        dnn.set_up_data_generator()

        # Model architecture definition and compilation
        dnn.define_model_architecture()

        # Train the DNN
        h = dnn.train(epochs=500)

        # Plot metrics from training when finished
        plot_metrics_legend(history=h, case=args.dnn)

        # Predict on some images to test the functionality of
        # dnn.predict()
