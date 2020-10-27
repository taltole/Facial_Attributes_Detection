import matplotlib.pyplot as plt
import numpy as np
from Classes.Train import *
from sklearn.metrics import classification_report, confusion_matrix


class Metrics:
    """
    This class contains a function that prints accuracy and loss graphs, confusion matrix and classification report 
    for a model
    """

    def __init__(self, history, y_test, y_pred):
        """
        :param filepath: path to the folder
        """
        self.history = history
        self.y_pred = y_pred
        self.y_test = y_test

    def acc_loss_graph(self):
        accuracy = self.history.history['accuracy']
        val_accuracy = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.plot(loss, 'r', label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(10), rotation='vertical')
        plt.title('train and val loss as function of epoch')
        plt.legend()
        plt.show()

        plt.plot(accuracy, 'r', label='training accuracy')
        plt.plot(val_accuracy, label='validation accuracy')
        plt.xlabel('# epochs')
        plt.xticks(np.arange(10), rotation='vertical')
        plt.ylabel('accuracy')
        plt.title('train and val accuracy as function of epoch')
        plt.legend()
        plt.show()

    def confusion_matrix(self):
        sns.heatmap(confusion_matrix(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1)), annot=True)
        plt.title('confusion matrix heatmap')
        plt.show()

    def classification_report(self):
        cr = classification_report(self.y_test.argmax(axis=1), self.y_pred.argmax(axis=1))
        print(cr)
