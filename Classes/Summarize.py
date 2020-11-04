import matplotlib.pyplot as plt
import numpy as np
from Classes.Train import *
from sklearn.metrics import classification_report, confusion_matrix
from config import *


class Metrics:
    """
    This class contains a function that prints accuracy and loss graphs, confusion matrix and classification report
    for a model
    """
    def __init__(self, history, epoch, y_test, y_pred, model_name, label):
        """
        :param history: json file
        :param epoch: nb of epoch
        :param y_test: list
        :param y_pred: list
        :param model_name: model name
        :param label: label name
        """
        self.history = history
        self.y_pred = y_pred
        self.y_test = y_test
        self.epoch = epoch
        self.label = label
        self.model_name = model_name

    def confusion_matrix(self):
        """
        Plot confusion matrix and save it in a csv file
        """
        print('Confusion Matrix ...')
        cm = confusion_matrix(self.y_test, self.y_pred)
        df = pd.DataFrame(cm)
        df.to_csv('csv/cm/' + self.model_name + self.label + '_cm.csv')
        print(df)

    def classification_report(self):
        """
        Plot classification report and save it in a csv file
        """
        print('Classification Report ...')
        cr = classification_report(self.y_test, self.y_pred, output_dict=True)
        df = pd.DataFrame(cr)
        df.to_csv('csv/cr/' + self.model_name + self.label + '_cr.csv')
        print(cr)

    def acc_loss_graph(self):
        """
        Plot Loss and Accuracy graphs
        """
        acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Train')
        plt.plot(val_acc, label='Val')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Train')
        plt.plot(val_loss, label='Val')
        plt.legend(loc='lower right')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim([0, max(plt.ylim())])
        plt.title('Loss')
        plt.show();
