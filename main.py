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

    # Argument Parser flags
    parser = argparse.ArgumentParser(
        description='Train a convolutional neural network')
    parser.add_argument(
        '--train_path', type=str, default='./datasets/non_realtime_dataset/train', help='Path to training dir')
    parser.add_argument(
        '--val_path', type=str, default='./datasets/non_realtime_dataset/validation', help='Path to validation dir')
    parser.add_argument(
        '--dnn', type=str, default='MCNA-v1-150x150', help='String with the name of the deep neural network')
    parser.add_argument(
        '--load_model', type=str, default='NO', help='String that indicates if a model wants to be loaded instead of training it again')
    parser.add_argument(
        '--detect_on_camera', type=str, default='NO',
        help='String that determines if the detection is performed processing a video or with a camera filming in real time')
    parser.add_argument(
        '--detect_video', type=str, default='NO', help='String that indicates if we want to generate a detected video')
    parser.add_argument(
        '--sliding_windows', type=str, default='NO', help='String that indicates if we want to execute sliding windows algorithm with te selected dnn')
    parser.add_argument(
        '--real_time', type=str, default='NO', help='String that determines if we want to execute or train the real time solution')

    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path
    input_size = None
    dnn = None
    model_ver = None

    # Instantiate a custom DNN to train and set up variables
    if args.dnn == 'MCNA-v1-150x150':
        input_size = (150, 150)
        dnn = MultiClassClassifierNoAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v1'
    elif args.dnn == 'MCNA-v2-150x150':
        input_size = (150, 150)
        dnn = MultiClassClassifierNoAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v2'

    elif args.dnn == 'MCA-v1-150x150':
        input_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v1'
    elif args.dnn == 'MCA-v1-224x224':
        input_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v1'
    elif args.dnn == 'MCA-v2-150x150':
        input_size = (150, 150)
        dnn = MultiClassClassifierAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v2'
    elif args.dnn == 'MCA-v2-224x224':
        input_size = (224, 224)
        dnn = MultiClassClassifierAugmentation(t_path=train_path, v_path=val_path, i_size=input_size)
        model_ver = 'v2'

    elif args.dnn == 'VGG16':
        input_size = (224, 224)
        dnn = MultiClassClassifierVGG16(t_path=train_path, v_path=val_path, i_size=input_size, num_classes=10)

    elif args.dnn == 'InceptionV3_TL':
        input_size = (150, 150)
        dnn = MultiClassClassifierInceptionV3TransferLearning(t_path=train_path, v_path=val_path, i_size=input_size)

    elif args.dnn == 'ResNet50_TL':
        input_size = (224, 224)
        dnn = MultiClassClassifierResNet50TransferLearning(t_path=train_path, v_path=val_path, i_size=input_size)

    elif args.dnn == 'RetinaNet_TL_FT':
        train_path = './datasets/realtime_dataset/train'
        annotations_path = './datasets/realtime_dataset/annotations'
        dnn = RealTimeClassifier(t_path=train_path, a_path=annotations_path)

    else:
        raise ValueError('Wrong DNN type :(')

    # Train or Load model
    if dnn is not None:
        if args.load_model == 'NO':  # Training case
            # Set up training data
            dnn.set_up_data()

            # Model architecture definition and compilation
            if model_ver == 'v1':
                dnn.define_model_architecture_v1()
            elif model_ver == 'v2':
                dnn.define_model_architecture_v2()
            else:
                dnn.compile_model()

            # Train the DNN
            if model_ver is not None:
                h = dnn.train(architecture_ver=model_ver)
            else:
                h = dnn.train()

            if args.dnn != 'RetinaNet_TL_FT':  # For the real time detector the metrics are different
                # Load the model to see the best loss and accuracy not the last ones
                if model_ver is None:
                    dnn.load_model()
                else:  # Select model version
                    dnn.load_model(av=model_ver)

                # Evaluate the model to get the loss value and the metrics values of the model in validation
                loss, accuracy = dnn.evaluate()

                # Plot metrics from training when finished
                plot_metrics_legend(h, args.dnn, loss, accuracy)

        elif args.load_model == 'YES':  # Loading model case
            if args.dnn != 'RetinaNet_TL_FT':
                dnn.set_up_data()
            if model_ver is None:
                dnn.load_model()
            else:  # Select model version
                dnn.load_model(av=model_ver)

        else:
            raise ValueError(' Please, you should only write YES or NO in the load_model option')

        if args.detect_on_camera == 'YES':  # Real time detection on camera
            if args.dnn == 'RetinaNet_TL_FT':
                dnn.detect_on_camera()
            else:
                raise ValueError('The selected model does not support real time detection on camera')

        elif args.detect_video == 'YES':  # Processing video with sliding windows or with real time detector
            # Prepare the video frames directory
            frames_dir = './results/video_processing/video_frames/'
            if os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir)
            os.mkdir(frames_dir)
            print('test_video directory ready for processing')

            # Start capturing the video and save the frames with jpg format into the video_frames directory
            video = cv2.VideoCapture('video_input.mp4')
            print('Processing video frames! wait few minutes...')
            success, image = video.read()
            count = 0
            while success:
                cv2.imwrite(frames_dir + '%d.jpg' % count, image)  # save frame as JPG file
                success, image = video.read()
                count += 1
            print('Finished processing video frames!')

            directory = None
            # Select if you want to detect using sliding windows algorithm or real time detection
            if args.sliding_windows == 'YES' and args.dnn != 'RetinaNet_TL_FT':
                directory = sliding_windows(args.dnn, dnn, input_size, frames_dir)
            elif args.real_time == 'YES':
                directory = dnn.predict(frames_dir)

            if directory is not None:
                listed_results_dir = os.listdir(directory)
                sorted_res_path = map(lambda l: int(l[:-4]), listed_results_dir)
                sorted_res_path = list(sorted_res_path)
                sorted_res_path.sort()

                size = (1920, 1080)  # FullHD
                output_video = cv2.VideoWriter(directory + 'resulting_video/resulting video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

                it = 0
                total = len(os.listdir(directory))
                if total != 0:
                    progress_bar(it, total, prefix='Assembling video frames: ', suffix='Complete', length=50)
                for frame in sorted_res_path:
                    it += 1
                    progress_bar(it, total, prefix='Assembling video frames: ', suffix='Complete', length=50)
                    f = cv2.imread(directory + str(frame) + '.jpg')
                    output_video.write(f)

                output_video.release()
                print('The video is finished! See the results playing the file: detected_video.avi')
            else:
                print('There was some error during the detection process, try changing model architecture')

        else:  # Evaluate model
            if args.dnn == 'RetinaNet_TL_FT':
                loss, accuracy = dnn.evaluate()

                # Code to save the evaluation results on a txt file inside evaluations directory
                file = open('./evaluations/evaluation_' + args.dnn + '.txt', 'a')
                file.write('[' + str(args.dnn) + '] -> Loss: ' + str(loss) + ' | Accuracy: ' + str(accuracy) + '\n')
                file.close()
            else:
                raise ValueError('Real time detector cannot be evaluated')

    else:
        raise ValueError('The dnn failed during initialization and is None')