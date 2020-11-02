import json
from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from config import *
from Classes.Train import *
from Classes.Summarize import *
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam


def main():

    # Start images processing and dataframe splitting
    trainer = Train(IND_FILE, IMAGEPATH)
    print('Reading File...\nCreating Train, Test...')
    label = 'Eyeglasses' # , 'Wearing_Hat', 'Wearing_Earrings']
    print(label)
    train, test = trainer.data_preprocess(IND_FILE, label, 10, True, None)
    print('Done!')

    # print('Checking test sample images...')
    # trainer.sanity_check(test)

    # Loading Base Model
    print(f'\nLoading Model...')
    model_list = ['vgg19', 'MobileNetV2', 'vgg_face', 'facenet', 'emotion', 'age', 'gender', 'race']
    print('Pick a Model: vgg19, MobileNetV2, vgg_face, facenet, emotion, age, gender, race')
    model_name = 'facenet'  # input('Choose one model to load: )

    # Training

    basemodel = BaseModel(model_name)

    model = basemodel.load_model(True)

    print(f'\nSave embedding...')

    feature_train, label_train = basemodel.loading_embedding(imagepath, model, train, 1)
    feature_test, label_test = basemodel.loading_embedding(imagepath, model, test, 1)

    print(feature_train)
if __name__ == '__main__':
    main()
