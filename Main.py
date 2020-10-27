from Classes import Summarize, LoadModel
from Classes.LoadModel import BaseModel
# from Classes.Summarize import Metrics
from config import *
from Classes.Train import *
from Classes.Summarize import *


def main():

    imagepath = IMAGE_PATH  # os.path.join(FACEPATH, '1')
    indexfile_path = '/Users/tal/Google Drive/Cellebrite/files list.csv'

    # Start images processing and dataframe splitting
    trainer = Train(indexfile_path, imagepath)
    print('Reading File...\nCreating Train, Test...')
    label = 'Eyeglasses'
    train, test = trainer.data_preprocess(IND_FILE, label, 5000, True, 224)
    print('Done!')

    # print('Checking test sample images...')
    # trainer.sanity_check(test)

    # Loading Base Model
    print(f'\nLoading Model...')
    print('Pick a Model: vgg19, MobileNetV2, vgg_face, facenet, emotion, age, gender, race')
    model_name = 'vgg19'  # input('Choose one model to load: )
    base_model = BaseModel(model_name)
    model = base_model.load_model(True)
    print(f'Model {model_name} loaded')

    # Split Train, Validation and Test Sets
    print(f'\nRunning data generator...')
    train_data, valid_data, test_data = trainer.generator_splitter(train, test, imagepath)

    # Training
    print(f'\nTraining Start...')
    model = base_model.adding_toplayers(model)
    history = trainer.start_train(model, label + '.h5', train_data, valid_data, 1, callback=None, optimize=None)
    print('Done!')

    metrics = Metrics(history, test_data)
    metrics.acc_loss_graph()

    # EPOCH = 1
    # STEP_SIZE_TEST = test_data.n // test_data.batch_size
    # loss, acc = model.evaluate(valid_data, steps=STEP_SIZE_TEST)
    # print(f"Loss:\t{round(loss, 2)}\nAcc.:\t{round(acc, 2)}")


if __name__ == '__main__':
    main()