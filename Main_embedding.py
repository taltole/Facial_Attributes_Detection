import json
from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from config import *
from Classes.Train import *
from Classes.Summarize import *
from Classes.BaseCls import *
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam


def main():
    # Start images processing and dataframe splitting
    trainer = Train(IND_FILE, IMAGE_PATH)
    print('Reading File...\nCreating Train, Test...')
    label = 'Eyeglasses'  # , 'Wearing_Hat', 'Wearing_Earrings']
    print(label)
    train, test = trainer.data_preprocess(IND_FILE, label, 40, True, None)
    print('Done!')

    # Loading Base Model
    print(f'\nLoading Model...')
    model_list = ['vgg19', 'MobileNetV2', 'vgg_face', 'facenet', 'emotion', 'age', 'gender', 'race']
    print('Pick a Model: vgg19, MobileNetV2, vgg_face, facenet, emotion, age, gender, race')
    model_name = 'facenet'  # input('Choose one model to load: )

    # Training
    basemodel = BaseModel(model_name)
    model = basemodel.load_model(True)

    print(f'\nSave embedding...')
    feature_train, label_train = basemodel.loading_embedding(IMAGE_PATH, model, train, 1)
    feature_test, label_test = basemodel.loading_embedding(IMAGE_PATH, model, test, 1)

    print('Running Grid Search on Cls...')
    df = gridsearch_cls(feature_train, label_train, feature_test, label_test)
    print(df)
    name_best_model = df['MLA Name'].values[0]
    print(name_best_model)

    # Plot best model
    i = 1
    ax = plt.subplot(1, 2, i)
    sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=df, color='m', ax=ax)
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.ylabel('Algorithm')

    # plot confusion matrix and acc score
    i += 1
    plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, i)
    cm = confusion_matrix(label_test, df['MLA pred'].values[0]) / len(label_test)
    accuracy = accuracy_score(label_test, df['MLA pred'].values[0])
    ax = sns.heatmap(cm, annot=True, cmap='Wistia', ax=ax)
    plt.title(f'{name_best_model}\n\nAccuracy: {accuracy * 100:.2f}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


if __name__ == '__main__':
    main()
