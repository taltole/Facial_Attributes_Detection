import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Layer, Input, Dropout, Activation, Convolution2D, MaxPooling2D

from BaseModels import Facenet, VGGFace
from AttModels import Age, Emotion, Gender, Race
from Classes.Face_preprocess import BasicFunction
from config import *
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_MNV2
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as Preprocess_RESNET50
from tensorflow.keras.preprocessing import image


class BaseModel:

    def __init__(self, model_name, include_top=False):
        """
        :param model_name: vgg19, vgg16, ResNet50, MobileNetV2,  vggface, facenet, emotion, age, gender, race
        :param include_top: include top layer of the model - Default value is False
        """
        self.model_name = model_name
        self.include_top = include_top

    @staticmethod
    def print_summary(model):
        """
        Print model summary
        :param model: model
        """
        print(f"Input_shape:\t{model.input_shape}\nOutput_shape:\t{model.output_shape}\nParams:\t{model.count_params()}"
              f"\nLayer:\t{len(model.layers)}\n\n")
        return model.summary()

    # Load Facial att. Models
    def load_model(self, include_top=False):
        """
        Load the model and print summary
        :param include_top: include top layer of the model - Default value is False
        """
        models = {'vggface': VGGFace,
                  'facenet': Facenet,
                  "emotion": Emotion,
                  "age": Age,
                  "gender": Gender,
                  "race": Race}

        if self.model_name == 'vgg19':
            base_model = tf.keras.applications.vgg19.VGG19(include_top=include_top, input_shape=(224, 224, 3))
        elif self.model_name == 'vgg16':
            base_model = tf.keras.applications.vgg16.VGG16(include_top=include_top, input_shape=(224, 224, 3))
        elif self.model_name == 'MobileNetV2':
            base_model = tf.keras.applications.MobileNetV2(include_top=include_top, input_shape=(224, 224, 3))
        elif self.model_name == 'ResNet50':
            base_model = tf.keras.applications.resnet50.ResNet50(include_top=include_top, input_shape=(224, 224, 3))
        else:
            try:
                base_model = models[self.model_name].loadModel()
            except ValueError:
                print('No model found')

        # self.print_summary(base_model)
        return base_model

    # ######################    Transfer Learning   ######################

    # Adding new model `model` whose first layer is base model chosen by user with additional Layer
    # (from `tensorflow.keras.Layer`):

    def adding_toplayer(self, base_model, name):
        """
        Function takes basemodel and add top Layer
        """
        # classes = 2
        # base_model_output = Sequential()
        # base_model_output = Convolution2D(classes, (1, 1), name='predictions')(basemodel.Layer[-4].output)
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
        # model = Convolution2D(classes, (1, 1), name='predictions')(basemodel.Layer[-4].output)
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
        # -----------------------------------------------------
        # inputshape = model.output_shape
        # # model.add(Flatten())
        # model.add(Convolution2D(64, 3, padding='same', input_shape=(inputshape, 32, 32, 3)))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Convolution2D(32, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(32, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # To train our transfer learning model we will freeze the weights of the basemodel and only train
        # the added Layers.
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        if name == 'vgg19':
            pass
        elif name == 'vggface7':
            model = Sequential()
            model.add(base_model)
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(5, activation='softmax'))

        elif name == 'vggface1':
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.25))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Activation('relu'))
        else:
            model.add(Dropout(0.25))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            self.print_summary(model)
        return model

    def loading_embedding(self, imagepath, model, data, layer_num):
        """
        :param imagepath: path to the image folder
        :param model: model
        :param data: dataset
        :param layer_num: position of the layer starting from the end of the summary. Determine where to cut the model
        This function takes a model, cut the layer and save the embedding for each images in a dataset
        """
        model = Model(inputs=model.input, outputs=model.layers[-layer_num].output)
        list_x = []
        for img in data['files'].tolist():
            if self.model_name not in ['vgg19', 'MobileNetV2', 'vggface', 'vgg16']:
                img = image.load_img(os.path.join(imagepath, img), target_size=(160, 160))
            else:
                img = image.load_img(os.path.join(imagepath, img), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            if self.model_name == 'vgg19':
                x = preprocess_input_VGG19(x)
            elif self.model_name == 'MobileNetV2':
                x = preprocess_input_MNV2(x)
            elif self.model_name == 'vgg16':
                x = preprocess_input_VGG16(x)
            elif self.model_name == 'ResNet50':
                x = Preprocess_RESNET50(x)
            else:
                x = x.astype('float32') / 255.
            list_x.append(x)
        feature_x = np.vstack(list_x)
        label = data['label'].tolist()
        feature = model.predict(feature_x)

        return feature, label

    # def make_model(self, input_shape, num_classes):
    #
    #     inputs = Input(shape=input_shape)
    #
    #     # Image augmentation block
    #     x = data_augmentation(inputs)
    #
    #     # Entry block
    #     x = Layer.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    #     x = Layer.Conv2D(32, 3, strides=2, padding="same")(x)
    #     x = Layer.BatchNormalization()(x)
    #     x = Layer.Activation("relu")(x)
    #
    #     x = Layer.Conv2D(64, 3, padding="same")(x)
    #     x = Layer.BatchNormalization()(x)
    #     x = Layer.Activation("relu")(x)
    #
    #     previous_block_activation = x  # Set aside residual
    #
    #     for size in [128, 256, 512, 728]:
    #         x = Layer.Activation("relu")(x)
    #         x = Layer.SeparableConv2D(size, 3, padding="same")(x)
    #         x = Layer.BatchNormalization()(x)
    #
    #         x = Layer.Activation("relu")(x)
    #         x = Layer.SeparableConv2D(size, 3, padding="same")(x)
    #         x = Layer.BatchNormalization()(x)
    #
    #         x = Layer.MaxPooling2D(3, strides=2, padding="same")(x)
    #
    #         # Project residual
    #         residual = Layer.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
    #         x = Layer.add([x, residual])  # Add back residual
    #         previous_block_activation = x  # Set aside next residual
    #
    #     x = Layer.SeparableConv2D(1024, 3, padding="same")(x)
    #     x = Layer.BatchNormalization()(x)
    #     x = Layer.Activation("relu")(x)
    #
    #     x = Layer.GlobalAveragePooling2D()(x)
    #     if num_classes == 2:
    #         activation = "sigmoid"
    #         units = 1
    #     else:
    #         activation = "softmax"
    #         units = num_classes
    #
    #     x = Layer.Dropout(0.5)(x)
    #     outputs = Layer.Dense(units, activation=activation)(x)
    #     return Model(inputs, outputs)

# model = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)
