import json
from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from config import *
from Classes.Train import *
from Classes.Summarize import *
from Classes.BaseCls import *
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam


def main(label, cls=MLA, exp=True):

    # Start images processing and dataframe splitting
    trainer = Train(IND_FILE, IMAGE_PATH)
    print('Reading File...\nCreating Train, Test...')

    print(f'Train on {label} attribute')
    train, test = trainer.data_preprocess(IND_FILE, label, 5000, True, None)
    print('Done!')

    # Loading Base Model
    print(f'\nLoading Model...')
    model_list = ['vgg19', 'MobileNetV2', 'vggface', 'facenet', 'emotion', 'age', 'gender', 'race']
    model_name = 'facenet'  # input('Choose one model to load: )

    basemodel = BaseModel(model_name)
    model = basemodel.load_model(True)
    # if model_name == 'vggface':
    #     model = basemodel.adding_toplayer(model)

    # Save embedding
    print(f'\nSave Embedding...')
    X_train, y_train = basemodel.loading_embedding(IMAGE_PATH, model, train, 1)
    X_test, y_test = basemodel.loading_embedding(IMAGE_PATH, model, test, 1)
    data_emb = pd.DataFrame(np.vstack([X_train, X_test]))

    print('GridSearch top Cls...')
    df_cls = gridsearch_cls(X_train, y_train, X_test, y_test, cls)
    print(df_cls.iloc[:, :-1])
    # name_best_cls = df_cls['MLA Name'].values[0]

    # Plot top classifier
    if not exp:
        plot_best_model(df_cls)

    # Optimizing
    if cls != 'xgb':
        print('\nOptimizing Hyper Parameters...')
        top_cls = gridsearch_params(df_cls, X_train, y_train, 3)
        print('Final Test for Best Classifier...')
        df_top_cls = gridsearch_cls(X_train, y_train, X_test, y_test, top_cls)
        print(df_top_cls.iloc[:, :-1], '-'*50, sep='\n')
        best_models = [i for i in top_cls['param'] if str(i).startswith(df_top_cls['MLA Name'].values[0])]

        cls = str(best_models).strip('[]')
        cls_name = cls.split('(')[0]
        print(f'Best Model:\n{cls}\n', '-'*50)
        best_cls = cls
    else:
        df_top_cls = df_cls
        cls_name = 'XGB'

    y_pred = df_top_cls['MLA pred'].values[0]

    # Saving embedding and final results to file
    label_emb = pd.DataFrame({'y_test': pd.Series(y_test), 'y_pred': pd.Series(y_pred)})
    label_emb.to_csv('csv/data/label_'+model_name+'_'+label+'_'+cls_name+'.csv')
    data_emb.to_csv('csv/data/data_'+model_name+'_'+label+'_'+cls_name+'.csv')
    df_top_cls.to_csv('csv/data/sum_'+model_name+'_'+label+'_top3.csv')

    # plot confusion matrix and acc score
    if not exp:
        ax = plt.subplot(1, 1, 1)
        cm = confusion_matrix(y_test, y_pred) / len(y_test)
        accuracy = accuracy_score(y_test, df_top_cls['MLA pred'].values[0])
        sns.heatmap(cm, annot=True, cmap='Wistia', ax=ax)
        plt.title(f'{cls}\n\nAccuracy: {accuracy * 100:.2f}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.show()

    # print("Checking XGB best params:")
    # check_xgb(feature_train, label_train)

    # model = eval(cls)()
    # get_model_results(model, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    df = pd.read_csv(IND_FILE)
    cols = df.columns.tolist()
    accessories_label = [l for l in cols if l.startswith("Wearing")]
    labels = accessories_label
    for label in labels:
        main(label, MLA, True)
