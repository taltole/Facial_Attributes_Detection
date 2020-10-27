from BaseModels import VGGFace
import gdown
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from config import *


def loadModel():
    model = VGGFace.baseModel()

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights
    model_name = 'gender_model_weights.h5'
    model_path = os.path.join(WEIGHT_PATH, model_name)

    if not os.path.isfile(model_path):
        print("gender_model_weights.h5 will be downloaded...")

        url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
        output = model_path
        gdown.download(url, output, quiet=False)

    gender_model.load_weights(model_path)

    return gender_model

# --------------------------
