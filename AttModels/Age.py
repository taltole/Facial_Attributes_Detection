from BaseModels import VGGFace
import gdown
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from config import *


def loadModel():
    model = VGGFace.baseModel()

    # --------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    # --------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    # --------------------------

    # load weights

    model_name = 'age_model_weights.h5'
    model_path = os.path.join(WEIGHT_PATH, model_name)

    if not os.path.isfile(model_path):
        print("age_model_weights.h5 will be downloaded...")

        url = 'https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV'
        output = model_path
        gdown.download(url, output, quiet=False)

    age_model.load_weights(model_path)

    return age_model


# --------------------------

def findApparentAge(age_predictions):
    output_indexes = np.array([i for i in range(0, 101)])
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age
