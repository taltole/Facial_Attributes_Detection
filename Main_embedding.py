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

    # Loading Base Model
    model_list = ['vgg19', 'MobileNetV2', 'vggface', 'facenet', 'emotion', 'age', 'gender', 'race']
    model_name = 'facenet'
    print(f'\nLoading {model_name} Base Model...')
    basemodel = BaseModel(model_name)
    model = basemodel.load_model(True)

    # Save or Load Embedding
    emb_file = 'data_' + model_name + '_' + label + '.csv'
    emb_path = os.path.join(EMB_PATH, emb_file)
    print('Reading File...\nCreating Train, Test...')
    print(f'\nTrain on {label} attribute\n')

    if os.path.exists(emb_path):
        print('Load Saved Embedding...')
        df_temp = pd.read_csv(emb_path)
        X_train, X_test = df_temp.iloc[:0.8*len(df), 0], df_temp.iloc[0.8*len(df):, 1]
        y_train, y_test = df_temp.iloc[:0.8*len(df), 2], df_temp.iloc[0.8*len(df):, 3]
    else:
        # Start images processing and dataframe splitting
        trainer = Train(IND_FILE, IMAGE_PATH)
        train, test = trainer.data_preprocess(IND_FILE, label, 50, True, None)
        print('Done!')

        # Save embedding
        print(f'\nSave Embedding...')
        X_train, y_train = basemodel.loading_embedding(IMAGE_PATH, model, train, 1)
        X_test, y_test = basemodel.loading_embedding(IMAGE_PATH, model, test, 1)
        data_emb = pd.DataFrame(np.hstack(np.vstack([X_train, X_test]), np.vstack([y_train, y_test])))
        data_emb.to_csv(os.getcwd() + '/csv/data/data_' + model_name + '_' + label + '.csv')

    # GridSearch Classifiers
    if not isinstance(cls, str):
        print('GridSearch top Cls...')
        df_cls = gridsearch_cls(X_train, y_train, X_test, y_test, cls)
        print(df_cls.iloc[:, :-1])
        # name_best_cls = df_cls['MLA Name'].values[0]

        # Plot top classifier
        if not exp:
            plot_best_model(df_cls)

        # Optimizing
        print('\nOptimizing Hyper Parameters...')
        top_cls = gridsearch_params(df_cls, X_train, y_train, 3)
        # print('Final Test for Best Classifier...')
        df_top_cls = gridsearch_cls(X_train, y_train, X_test, y_test, top_cls)
        # print(df_top_cls.iloc[:, :-1], '-'*50, sep='\n')
        best_models = [i for i in top_cls['param'] if str(i).startswith(df_top_cls['MLA Name'].values[0])]
        cls = str(best_models).strip('[]')
        cls_name = cls.split('(')[0]
        print(f'Best Model:\n{cls}\n', '-' * 50)
        # best_cls = cls
        # y_pred = df_top_cls['MLA pred'].values[0]
    else:
        top_cls = cls
        df_top_cls = gridsearch_cls(X_train, y_train, X_test, y_test, top_cls)
        # best_models = [i for i in top_cls if str(i).startswith(df_top_cls['MLA Name'].values[0])]
        # cls = str(best_models).strip('[]')
        cls_name = cls
        print(f'Best Model:\n{cls}\n', '-' * 50)
        # best_cls = cls

    print('Final Test for Best Classifier...')
    print(df_top_cls.iloc[:, :-1], '-' * 50, sep='\n')
    y_pred = df_top_cls['MLA pred'].values[0]

    # Saving embedding and final results to file
    # label_emb = pd.DataFrame({'y_test': pd.Series(y_test), 'y_pred': pd.Series(y_pred)})
    # label_emb.to_csv(os.getcwd() + '/csv/data/label_' + model_name + '_' + label + '_' + cls_name + '.csv')
    df_top_cls.to_csv(os.getcwd() + '/csv/data/sum_' + model_name + '_' + label + '_top3.csv')

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
    # accessories_label = [l for l in cols if l.startswith("Wearing")]
    labels = ['Arched_Eyebrows',
              'Attractive',
              'Bags_Under_Eyes',
              'Bald',
              'Bangs',
              'Big_Lips',
              'Big_Nose',
              'Bushy_Eyebrows',
              'Chubby',
              'Double_Chin',
              'Eyeglasses',
              'Goatee',
              'Heavy_Makeup',
              'High_Cheekbones',
              'Mustache',
              'Narrow_Eyes',
              'No_Beard',
              'Oval_Face',
              'Pale_Skin',
              'Pointy_Nose',
              'Receding_Hairline',
              'Rosy_Cheeks',
              'Sideburns',
              'Smiling']
    for label in labels:
        main(label, MLA, True)
