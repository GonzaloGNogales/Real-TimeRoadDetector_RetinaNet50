import argparse
from MultiClassClassifier.multiclass_classifier_no_augmentation import *
from MultiClassClassifier.multiclass_classifier_augmentation import *
from MultiClassClassifier.multiclass_classifier_VGG16based_model import *
from MultiClassClassifier.multiclass_classifier_Inceptionv3_transferlearning import *
from MultiClassClassifier.multiclass_classifier_ResNet50_transferlearning import *
from DeepLearningUtilities.metrics_analyzer import *
from SlidingWindowsAlgorithm.sliding_windows import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train a convolutional neural network')
    parser.add_argument(
        '--train_path', type=str, default='./training/', help='Path to training dir')
    parser.add_argument(
        '--val_path', type=str, default='./validation/', help='Path to validation dir')
    parser.add_argument(
        '--dnn', type=str, default='MCNA', help='String referring to the neural network selection')
    parser.add_argument(
        '--load_model', type=str, default='NO', help='String that indicates if we want to load a model instead of training it')
    parser.add_argument(
        '--sliding_windows', type=str, default='NO', help='String that indicates if we want to execute sliding windows algorithm with te selected dnn')
    parser.add_argument(
        '--real_time', type=str, default='NO', help='String that determines if the code that will execute has to do with the fully convolutional solution')

    args = parser.parse_args()
    dnn = None
    model_ver = None
    spe = True
    i_size = None

    # Instantiate a custom DNN to train
    if args.dnn == 'MCNA':
        i_size = (150, 150)
        dnn = MultiClassClassifierNoAugmentation()  # Instantiate HERE
        model_ver = 'v1'
    elif args.dnn == 'MCA':
        dnn = MultiClassClassifierAugmentation()
        model_ver = 'v1'
    elif args.dnn == 'MCA-v1-150x150':
        i_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v1'
        spe = False
    elif args.dnn == 'MCA-v1-150x150-spe':
        i_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v1'
        spe = True
    elif args.dnn == 'MCA-v1-224x224':
        i_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v1'
        spe = False
    elif args.dnn == 'MCA-v1-224x224-spe':
        i_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v1'
        spe = True
    elif args.dnn == 'MCA-v2-150x150':
        i_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v2'
        spe = False
    elif args.dnn == 'MCA-v2-150x150-spe':
        i_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v2'
        spe = True
    elif args.dnn == 'MCA-v2-224x224':
        i_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v2'
        spe = False
    elif args.dnn == 'MCA-v2-224x224-spe':
        i_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(i_size=i_size)
        model_ver = 'v2'
        spe = True
    elif args.dnn == 'VGG16':
        i_size = (224, 224)
        dnn = MultiClassClassifierVGG16(num_classes=9)
    elif args.dnn == 'InceptionV3_TL':
        dnn = MultiClassClassifierInceptionV3TransferLearning()
        i_size = (150, 150)
    elif args.dnn == 'ResNet50_TL':
        dnn = MultiClassClassifierResNet50TransferLearning()
        i_size = (224, 224)
    else:
        raise ValueError('Wrong DNN type :(')

    if dnn is not None:

        # Set up data generator for pre processing the images and fixing batch size
        dnn.set_up_data_generator()

        if args.load_model != 'YES':  # Training case
            # Model architecture definition and compilation
            if model_ver == 'v1':
                dnn.define_model_architecture_v1()
            elif model_ver == 'v2':
                dnn.define_model_architecture_v2()
            else:
                dnn.compile_model()

            # Train the DNN
            if model_ver is not None:
                h = dnn.train(epochs=100, steps_per_epoch=spe, architecture_ver=model_ver)
            else:
                h = dnn.train(epochs=100)

            # Evaluate the model to get the loss value and the metrics values of the model in validation
            loss, accuracy = dnn.evaluate()

            # Plot metrics from training when finished
            plot_metrics_legend(h, args.dnn, loss, accuracy)
        elif args.load_model == 'YES':  # Evaluation case
            if args.dnn == 'VGG16':
                dnn.load_model_weights()
            elif args.dnn == 'InceptionV3_TL' or args.dnn == 'ResNet50_TL':
                dnn.load_model()
            else:
                dnn.load_model(spe=spe, av=model_ver)
            if args.sliding_windows != 'YES':
                loss, accuracy = dnn.evaluate()

                # Code to save the evaluation results on a txt file
                file = open("evaluation_TL.txt", "a")
                file.write('[' + str(args.dnn) + '] -> Loss: ' + str(loss) + ' | Accuracy: ' + str(accuracy) + '\n')
                file.close()

        if args.sliding_windows == 'YES':
            sliding_windows(args.dnn, dnn, input_size=i_size)
