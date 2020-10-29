import json
from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from config import *
from Classes.Train import *
from Classes.Summarize import *
import tensorflow as tf


def main():
    imagepath = IMAGE_PATH  # os.path.join(FACEPATH, '1')
    indexfile_path = IND_FILE

    # Start images processing and dataframe splitting
    trainer = Train(indexfile_path, imagepath)
    print('Reading File...\nCreating Train, Test...')
    label = 'Eyeglasses'
    train, test = trainer.data_preprocess(IND_FILE, label, 5000, True, 224)
    print('Done!')

    # print('Checking test sample images...')
    # trainer.sanity_check(test)

    # Split Train, Validation and Test Sets
    print(f'\nRunning data generator...')
    train_data, valid_data, test_data = trainer.generator_splitter(train, test, imagepath)

    # Loading Base Model
    print(f'\nLoading Model...')
    print('Pick a Model: vgg19, MobileNetV2, vgg_face, facenet, emotion, age, gender, race')
    model_name = 'vgg_face'  # input('Choose one model to load: )

    # Training
    print(f'\nTraining Start...')
    basemodel = BaseModel(model_name)

    model_file = model_name + '_' + label + '.h5'
    epoch = 4
    train = False

    if train:
        if model_name not in ['vgg19', 'MobileNetV2']:
            model = basemodel.load_model(False)
        else:
            model = basemodel.load_model()
        history, model = trainer.start_train(model, model_file, train_data, valid_data, epoch, callback=None,
                                             optimize=None)
        print('Loading best weights...')
        model.load_weights(model_file)
        print('Done!')

        # Saving History
        with open(model_name + '_' + label + '.json', 'w') as f:
            json.dump(history.history, f)
    else:
        history = json.load(open(model_name + '_' + label + '.json'))
        model = basemodel.load_model(False)
        model = basemodel.adding_toplayer(model)
        print(f'\nModel {model_name} Loaded!')

        print('Loading best weights...')
        model.load_weights(model_file)
        model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=["accuracy"])

    # Evaluate the network on valid data
    Prediction.evaluate_model(model, valid_data)

    # Predict on test data
    y_pred = Prediction.test_prediction(model, test_data, train_data)

    # plot
    top = min(len(test['label']), len(y_pred))
    metrics = Metrics(history, epoch, test['label'][:top].tolist(), y_pred[:top])
    metrics.confusion_matrix()
    metrics.acc_loss_graph()
    metrics.classification_report()

    # Inference
    labels = [test['files'][test['label'] == '1.0'], test['files'][test['label'] == '0.0']]
    pos, neg = f'With {label}', f'W/O {label}'
    Prediction.predict_label(model, labels, pos, neg)
    file = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/3/face_att_174563.jpg'
    Prediction.predict_file(model, file, pos, neg)


if __name__ == '__main__':
    main()
