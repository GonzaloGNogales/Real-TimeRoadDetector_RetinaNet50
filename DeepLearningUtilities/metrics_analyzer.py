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


def plot_metrics_legend(history, case, v_loss, v_acc):
    if history is not None:
        if not os.path.isdir('./metrics2/' + case):
            os.mkdir('./metrics2/' + case)

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(accuracy))

        accuracy_plot, ax_acc = plt.subplots()
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        props = dict(boxstyle='round', facecolor='green', alpha=0.5)
        ax_acc.text(0.95, 0.5, 'val_acc = {:.6f}'.format(v_acc), transform=ax_acc.transAxes, fontsize=12,
                    horizontalalignment='right',
                    verticalalignment='center', bbox=props)
        plt.legend()
        plt.savefig('./metrics2/' + case + '/' + 'Train and Validation Accuracy MultiClassClassifier.png')

        loss_plot, ax_loss = plt.subplots()
        plt.plot(epochs, loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        props = dict(boxstyle='round', facecolor='blue', alpha=0.5)
        ax_loss.text(0.95, 0.5, 'val_loss = {:.6f}'.format(v_loss), transform=ax_loss.transAxes, fontsize=12,
                     horizontalalignment='right',
                     verticalalignment='center', bbox=props)
        plt.legend()
        plt.savefig('./metrics2/' + case + '/' + 'Train and Validation Loss MultiClassClassifier.png')
