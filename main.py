import argparse
from MultiClassClassifier.multiclass_classifier_no_augmentation import *
from MultiClassClassifier.multiclass_classifier_augmentation import *
from MultiClassClassifier.multiclass_classifier_VGG16based_model import *
from MultiClassClassifier.multiclass_classifier_Inceptionv3_transferlearning import *
from MultiClassClassifier.multiclass_classifier_ResNet50_transferlearning import *
from RealTimeMultiLabelClassifier.realtime_classifier import *
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
        '--sliding_windows_test', type=str, default='NO', help='String that indicates if we want to execute sliding windows algorithm with te selected dnn in test mode')
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
    elif args.dnn == 'RetinaNet_TL_FN':
        dnn = RealTimeClassifier()
    else:
        raise ValueError('Wrong DNN type :(')

    if dnn is not None:

        # Set up data generator for pre processing the images and fixing batch size
        dnn.set_up_data()

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
                h = dnn.train()

            if args.dnn != 'RetinaNet_TL_FN':
                # Evaluate the model to get the loss value and the metrics values of the model in validation
                loss, accuracy = dnn.evaluate()

                # Plot metrics from training when finished
                plot_metrics_legend(h, args.dnn, loss, accuracy)

        elif args.load_model == 'YES':  # Evaluation case
            if args.dnn == 'VGG16' or args.dnn == 'InceptionV3_TL' or args.dnn == 'ResNet50_TL':
                dnn.load_model()
            else:
                dnn.load_model(spe=spe, av=model_ver)

            if args.sliding_windows != 'YES':
                loss, accuracy = dnn.evaluate()

                # Code to save the evaluation results on a txt file inside evaluations directory
                file = open('./evaluations/evaluation_' + args.dnn + '.txt', 'a')
                file.write('[' + str(args.dnn) + '] -> Loss: ' + str(loss) + ' | Accuracy: ' + str(accuracy) + '\n')
                file.close()

        if args.sliding_windows == 'YES':
            # Prepare the video testing directory
            test_dir = './test_video/'
            if os.path.isdir(test_dir):
                shutil.rmtree(test_dir)
            os.mkdir(test_dir)
            print('test_video directory ready for processing')

            # Start capturing the video and save the frames with jpg format into the test_video directory
            video = cv2.VideoCapture('video_input.mp4')
            print('Processing video frames! wait few minutes...')
            success, image = video.read()
            count = 0
            while success:
                cv2.imwrite(test_dir + '%d.jpg' % count, image)  # save frame as JPEG file
                success, image = video.read()
                count += 1
            print('Finished processing video frames!')

            # detector.preprocess_data('./test_video/', False)  sliding windows on ./test_video/
            directory = sliding_windows(args.dnn, dnn, i_size, test_dir)

            final_video_array = list()
            it = 0
            total = len(os.listdir(directory))
            if total != 0:
                progress_bar(it, total, prefix='Loading video frames: ', suffix='Complete', length=50)
            for frame in os.listdir(directory):
                it += 1
                progress_bar(it, total, prefix='Loading video frames: ', suffix='Complete', length=50)
                if os.path.isfile(directory + frame) and frame.endswith('.jpg'):
                    f = cv2.imread(directory + frame)
                    final_video_array.append((f, frame))
            h, w, _ = final_video_array[0][0].shape
            size = (w, h)
            output_video = cv2.VideoWriter(directory + 'detected_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

            # Progress bar for giving feedback to the users
            final_video_array.sort(key=lambda tup: int(tup[1][:-4]))
            it = 0
            total = len(final_video_array)
            if total != 0:
                progress_bar(it, total, prefix='Video mounting progress: ', suffix='Complete', length=50)
            for i in range(len(final_video_array)):
                it += 1
                progress_bar(it, total, prefix='Video mounting progress: ', suffix='Complete', length=50)
                output_video.write(final_video_array[i][0])
            output_video.release()
            print('The video is finished! See the results playing the file: detected_video.avi')

        elif args.sliding_windows_test == 'YES':
            test_dir = './images_sw/'
            sliding_windows(args.dnn, dnn, i_size, test_dir)
