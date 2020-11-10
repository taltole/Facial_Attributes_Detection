from BaseModels import VGGFace

import gdown
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
import zipfile
from config import *


def loadModel():
    model = VGGFace.baseModel()

    # --------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    model_name = 'race_model_weights.h5'
    model_path = WEIGHT_PATH + model_name

    if not os.path.isfile(model_path):
        print("race_model_single_batch.h5 will be downloaded...")

        # zip
        url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
        output = WEIGHT_PATH + '/race_model_single_batch.zip'
        gdown.download(url, output, quiet=False)

        # unzip race_model_single_batch.zip
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(WEIGHT_PATH)

    race_model.load_weights(model_path)

    return race_model

# --------------------------
