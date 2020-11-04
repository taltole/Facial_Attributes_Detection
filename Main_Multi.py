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

    model_name = 'vgg19'

    # Split Train, Validation and Test Sets
    print(f'\nRunning data generator...')
    train_data, valid_data, test_data = Multi.generator_splitter_multi(model_name, train, test, IMAGE_PATH)
    # print(train['image'].shape)
    # run_ensemble(train['image'], train['label'], test['image'], test['label'])

    # print('Checking test sample images...')
    # trainer.sanity_check(test)


    # Loading Base Model
    print(f'\n\nLoading Model...')
    print('Pick a Model: vgg19, MobileNetV2, vggface, facenet, emotion, age, gender, race')

    label_name = 'Hair_color'
    model_file = os.path.join('weights/', model_name + '_' + label_name + '.h5')
    json_path = os.path.join('json/', model_name + '_' + label_name + '.json')
    epoch = 2

    # Training
    print(f'\nTraining Start...')
    basemodel = BaseModel(model_name)

    training = True

    if training:
        model = basemodel.load_model()
        model = basemodel.adding_toplayer(model)
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
        history = json.load(open(json_path))
        model = basemodel.load_model(False)
        model = basemodel.adding_toplayer(model)
        print(f'\nModel {model_name} Loaded!')

        print('Loading best weights...')
        model.load_weights(os.path.join(WEIGHT_PATH, model_name + '_' + label_name + '.h5'))
        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # opt_list = {'lr': [0.001, 0.005, 0.0001, 0.0005], 'decay': [1e-6]}
        model.compile(RMSprop(lr=0.0001, decay=1e-6), loss='categorical_crossentropy', metrics=["accuracy"])

    # Evaluate the network on valid data
    # Prediction.evaluate_model(model, valid_data)

    # Predict on test data
    y_pred = Prediction.test_prediction(model, test_data, train_data)

    # plot
    top = min(len(test['label']), len(y_pred))
    metrics = Metrics(history, epoch, test['label'][:top].tolist(), y_pred[:top], model_name, label_name)
    metrics.confusion_matrix()
    metrics.acc_loss_graph()
    metrics.classification_report()

        # Inference
    # labels = [test['files'][test['label'] == '1.0'], test['files'][test['label'] == '0.0']]
    # pos, neg = f'With {label}', f'W/O {label}'
    # Prediction.predict_label(model, labels, pos, neg)
    # file = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/3/face_att_174563.jpg'
    # Prediction.predict_file(model, file, pos, neg)


"""
    # layer_name = 'my_dense'
    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer(layer_name).output)

    # intermediate_layer_model.summary()
    model.summary()

    # intermediate_output = intermediate_layer_model.predict()
    intermediate_output = y_pred
    intermediate_output = pd.DataFrame(data=intermediate_output)

    # val_data = intermediate_output[53000:]

    submission_cnn = model.predict(np.float32(test['image']))

    # intermediate_test_output = intermediate_layer_model.predict(test['image'])
    intermediate_test_output = model.predict(np.float32(test['image']))
    intermediate_test_output = pd.DataFrame(data=intermediate_test_output)

    # xgbmodel = XGBClassifier(objective='multi:softprob', num_class=2)
    # xgbmodel.fit(intermediate_output, train_label1)
    # xgbmodel.score(val_data, val_label1)
    #
    # intermediate_layer_model.predict(X_test)
    # submission_xgb = xgbmodel.predict(intermediate_test_output)

    from sklearn.naive_bayes import GaussianNB

    gnbmodel = GaussianNB().fit(intermediate_output, np.float32(train['label']))

    submission_gnb = gnbmodel.predict(intermediate_test_output)
    gnbmodel.score(np.float32(train['image'][valid_data.filenames]), valid_data.labels)

    submission_cnn = submission_cnn.astype(int)

    label = np.argmax(submission_cnn, 1)
    id_ = np.arange(0, label.shape[0])
    print(label)

"""


if __name__ == '__main__':
    main()