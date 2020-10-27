import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from BaseModels import Facenet, VGGFace
from AttModels import Age, Emotion, Gender, Race
from Classes.Face_preprocess import BasicFunction
from config import *
# from deepface.commons import functions, realtime, distance as dst


# ######################    Modeling   ######################


class BaseModel:
    def __init__(self, model_name, include_top=False, builtin_preprocess=False):
        self.model_name = model_name
        self.include_top = include_top
        if builtin_preprocess:
            self.preprocess = BasicFunction(IMAGEPATH).preprocessing(img_size=224)

    @staticmethod
    def print_summary(model):
        print(f"Input_shape:\t{model.input_shape}\nOutput_shape:\t{model.output_shape}\nParams:\t{model.count_params()}"
              f"\nLayers:\t{len(model.layers)}\n\n")
        return model.summary()

    # Load FaceDetection Models
    def load_model(self, include_top=False):
        """
        load face detection models
        """
        models = {'vgg_face': VGGFace,
                  'facenet': Facenet,
                  "emotion": Emotion,
                  "age": Age,
                  "gender": Gender,
                  "race": Race}

        if self.model_name == 'vgg19':
            base_model = tf.keras.applications.vgg19.VGG19(include_top, input_shape=(224, 224, 3))
        elif self.model_name == 'MobileNetv2':
            base_model = tf.keras.applications.MobileNetV2(include_top, input_shape=(224, 224, 3))
        else:
            try:
                base_model = models[self.model_name].loadModel()
            except:
                print('No model found')

        self.print_summary(base_model)
        return base_model

    # ######################    Transfer Learning   ######################

    # Adding new model `model` whose first layer is base model chosen by user with additional layers
    # (from `tensorflow.keras.layers`):

    def adding_toplayers(self, base_model):
        """
        Function takes basemodel and add top layers
        """
        # classes = 2
        # base_model_output = Sequential()
        # base_model_output = Convolution2D(classes, (1, 1), name='predictions')(basemodel.layers[-4].output)
        # base_model_output = Flatten()(base_model_output)
        # base_model_output = Dense(128, activation='relu')(base_model_output)
        # base_model_output = Dense(2, activation='relu')(base_model_output)
        # base_model_output = Activation('relu')(base_model_output)
        # model = Model(inputs=vggface.input, outputs=base_model_output)
        # -----------------------------------------------------------
        # model = Sequential()
        # model.add(basemodel)
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # ------------------------------------------------------------
        # model = Sequential()
        # model = Convolution2D(classes, (1, 1), name='predictions')(basemodel.layers[-4].output)
        # model = Flatten()(model)
        # model = Activation('relu')(model)
        # model = Flatten()(model)
        # model = Convolution2D(64, 3, padding='same', input_shape=(32,32,3))(model)
        # model = Activation('relu')(model)
        # model = Convolution2D(64, (3, 3))(model)
        # model = Activation('relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        # model = Dropout(0.25)(model)
        # model = Convolution2D(32, (3, 3), padding='same')(model)
        # model = Activation('relu')(model)
        # model = Convolution2D(32, (3, 3))(model)
        # model = Activation('relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        # model = Dropout(0.25)(model)
        # model = Flatten()(model)
        # model = Dense(512)(model)
        # model = Activation('relu')(model)
        # model = Dropout(0.5)(model)
        # model = Dense(10, activation='relu')(model)

        # To train our transfer learning model we will freeze the weights of the basemodel and only train the added layers.

        base_model.trainable = False
        model = Sequential()
        model.add(base_model)

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        self.print_summary(model)
        return model


# def main():
#     # VGGFace.loadModel()
#
#
# if __name__ == '__main__':
#     main()