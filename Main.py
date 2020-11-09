from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
from Classes.Summarize import *
from tensorflow.keras.optimizers import RMSprop, SGD, Adam


def main(label):

    # Start images processing and dataframe splitting
    trainer = Train(IND_FILE, IMAGE_PATH)
    print('Reading File...')
    print(f'Preparing data for class:\t{label}\nCreating Train, Test...')
    train, test = trainer.data_preprocess(IND_FILE, label, 5000, True, None)
    print('Done!')

    # print('Checking test sample images...')
    # trainer.sanity_check(test)

    # Split Train, Validation and Test Sets
    print(f'\nRunning data generator...')
    train_data, valid_data, test_data = trainer.generator_splitter(train, test, IMAGE_PATH)

    # Loading Base Model
    print(f'\n\nLoading Model...')
    model_list = ['vgg19', 'vgg16', 'MobileNetV2', 'vggface', 'facenet']  # , 'emotion', 'age', 'gender', 'race']
    print('Pick a Model: vgg19, vgg16, ResNet50, MobileNetV2,  vggface, facenet, emotion, age, gender, race')

    # Looping over models
    for model_name in model_list:
        # model_name  = 'vgg_face'  # input('Choose one model to load: )
        model_file = os.path.join('weights/', model_name + '_' + label + '_L2.h5')
        json_path = os.path.join('json/', model_name + '_' + label + '_L2.json')
        epoch = 100

        # Training
        print(f'\nTraining Start...')
        basemodel = BaseModel(model_name)

        training = True

        if training:
            model = basemodel.load_model()
            model = basemodel.adding_toplayer(model)
            history, model = trainer.start_train(model, model_file, train_data, valid_data, epoch, multi=False,
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
            history = json.load(open(json_path))
            model = basemodel.load_model(False)
            model = basemodel.adding_toplayer(model)
            print(f'\nModel {model_name} Loaded!')

            print('Loading best weights...')
            model.load_weights(os.path.join(WEIGHT_PATH, model_name + '_' + label + '.h5'))
            # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # opt_list = {'lr': [0.001, 0.005, 0.0001, 0.0005], 'decay': [1e-6]}
            model.compile(RMSprop(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=["accuracy"])

        # Evaluate the network on valid data
        Prediction.evaluate_model(model, valid_data)

        # Predict on test data
        y_pred = Prediction.test_prediction(model, test_data, train_data)

        # plot
        top = min(len(test['label']), len(y_pred))
        metrics = Metrics(history, epoch, test['label'][:top].tolist(), y_pred[:top], model_name, label)
        metrics.confusion_matrix()
        #metrics.acc_loss_graph()
        metrics.classification_report()


if __name__ == '__main__':
    
    df = pd.read_csv(IND_FILE)
    cols = df.columns.tolist()
    accessories_label = [l for l in cols if l.startswith("Wearing")]
    # hair_label = [l for l in cols if "Hair" in l and not l.startswith('0')]
    labels = accessories_label

    for label in labels:
        main(label)
