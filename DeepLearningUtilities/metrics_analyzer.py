import os
import matplotlib.pyplot as plt


def plot_metrics_no_legend(history):
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(accuracy))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, accuracy, 'r', "Training Accuracy")
    plt.plot(epochs, val_accuracy, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")

    plt.title('Training and validation loss')
    plt.show()


def plot_metrics_legend(history, case):
    if history is not None:
        if not os.path.isdir('./metrics/' + case):
            os.mkdir('./metrics/' + case)

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(accuracy))

        accuracy_plot = plt.figure()
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        accuracy_plot.savefig('./metrics/' + case + '/' + 'Trainig and Validation Accuracy MultiClassClassifier No Augmentation.png')

        loss_plot = plt.figure()
        plt.plot(epochs, loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()
        loss_plot.savefig('./metrics/' + case + '/' + 'Trainig and Validation Loss MultiClassClassifier No Augmentation.png')
