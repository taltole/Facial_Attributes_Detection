from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from Classes.Summarize import *
from Classes.Multiclass_model import *
from tensorflow.keras.optimizers import RMSprop, SGD, Adam


def main():
    # Start images processing and dataframe splitting
    Multi = Multiclass_Model(IND_FILE)
    trainer = Train(IND_FILE, IMAGE_PATH)
    print('Reading File...')
    # label = 'Eyeglasses'  # , 'Wearing_Hat', 'Wearing_Earrings']
    print(f'Preparing data.. \nCreating Train, Test...')

    label_list = ['Brown_Hair','Blond_Hair', 'Black_Hair','Bald', 'Gray_Hair']
    train, test = Multi.create_dataframe_multi(label_list, 100)
    print('Done!')

    print('Pick a Model: vgg19, vgg16, ResNet50, MobileNetV2, vggface, facenet, emotion, age, gender, race')
    model_name = 'vgg19'

    # Split Train, Validation and Test Sets
    print(f'\nRunning data generator...')
    train_data, valid_data, test_data = Multi.generator_splitter_multi(model_name, train, test, IMAGE_PATH)


    label_name = 'Hair_color'
    model_file = os.path.join('weights/', model_name + '_' + label_name + '.h5')
    json_path = os.path.join('json/', model_name + '_' + label_name + '.json')
    epoch = 2

    basemodel = BaseModel(model_name)

    training = True

    if training:
        print(f'\n\nLoading Model...')
        model = basemodel.load_model()
        model = basemodel.adding_toplayer(model)
        print(f'\nTraining Start...')
        history, model = trainer.start_train(model, model_file, train_data, valid_data, epoch, multi=True,
                                             callback=None,
                                             optimize=None)
        print('Loading best weights...')
        model.load_weights(model_file)
        print('Done!')

        # Saving History
        with open(json_path, 'w') as f:
            json.dump(history.history, f)
        history = json.load(open(json_path))

    else:
        print(f'\n\nLoading Model...')
        history = json.load(open(json_path))
        model = basemodel.load_model(False)
        model = basemodel.adding_toplayer(model)
        print(f'\nModel {model_name} Loaded!')

        print('Loading best weights...')
        model.load_weights(os.path.join(WEIGHT_PATH, model_name + '_' + label_name + '.h5'))
        model.compile(RMSprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=["accuracy"])

    # Evaluate the network on valid data
    Prediction.evaluate_model(model, valid_data)

    # Predict on test data
    y_pred = Prediction.test_prediction(model, test_data, train_data)

    # plot
    top = min(len(test['label']), len(y_pred))
    metrics = Metrics(history, epoch, test['label'][:top].tolist(), y_pred[:top], model_name, label_name)
    metrics.confusion_matrix()
    metrics.acc_loss_graph()
    metrics.classification_report()

    labels_hair = {'Bald': 0, 'Black_Hair': 1, 'Blond_Hair': 2, 'Brown_Hair': 3, 'Gray_Hair': 4}
    Prediction.predict_label_multi(model, labels_hair, IMAGE_PATH + '/' + 'face_att_018217.jpg', 'ResNet50')


if __name__ == '__main__':
    main()
