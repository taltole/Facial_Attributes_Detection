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
    model_list = ['vgg19', 'MobileNetV2', 'vggface', 'facenet', 'emotion', 'age', 'gender', 'race']
    model_name = 'facenet'  # input('Choose one model to load: )

    # Training
    basemodel = BaseModel(model_name)
    model = basemodel.load_model(True)

    #Save embedding
    print(f'\nSave embedding...')
    feature_train, label_train = basemodel.loading_embedding(IMAGE_PATH, model, train, 1)
    feature_test, label_test = basemodel.loading_embedding(IMAGE_PATH, model, test, 1)

    print('Running Grid Search on Cls...')
    df_cls = gridsearch_cls(feature_train, label_train, feature_test, label_test, MLA)
    plot_best_model(df_cls)
    print(df_cls)

    name_best_model = df_cls['MLA Name'].values[0]

    print('Starting Hyper_parameters GridSearch...')
    top_cls = gridsearch_params(df_cls, feature_train, label_train)
    print(top_cls['param'])

    df_top_cls = gridsearch_cls(feature_train, label_train, feature_test, label_test, top_cls)
    print(df_top_cls)
    best_model = [i for i in top_cls['param'] if str(i).startswith(df_top_cls['MLA Name'].values[0])]

    cls = str(best_model)
    # plot confusion matrix and acc score
    ax = plt.subplot(1, 1, 1)
    cm = confusion_matrix(label_test, df_top_cls['MLA pred'].values[0]) / len(label_test)
    accuracy = accuracy_score(label_test, df_top_cls['MLA pred'].values[0])
    sns.heatmap(cm, annot=True, cmap='Wistia', ax=ax)
    plt.title(f'{name_best_model}\n\nAccuracy: {accuracy * 100:.2f}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    # print("Checking XGB best params:")
    # check_xgb(feature_train, label_train)

    # get_model_results(best_model, feature_train, feature_test, label_train, label_test)


def plot_best_model(df):
    plt.figure(figsize=(16, 7))
    ax = plt.subplot(1, 1, 1)
    sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=df, color='m', ax=ax)
    plt.title('Machine Learning Algorithm Accuracy Score \n')
    plt.xlabel('Accuracy Score (%)')
    plt.xticks(np.arange(0, 1, 0.1))
    plt.ylabel('Algorithm')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
