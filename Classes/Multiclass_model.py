import pandas as pd
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Multiclass_Model:
    def __init__(self, index_file):
        self.index_file = index_file

    def data_preprocess_multi(self, label, nb_data):
        # for color in list(dict_color.keys()):
        df = pd.read_csv(self.index_file, usecols=[label])
        df_label = df[(df[label] != 0) & (df[label] != '0')][:nb_data]
        train_size = int(len(df_label) * 0.8)

        train = df_label[:train_size]
        labels_train = [label] * train_size
        train = pd.DataFrame(
            {'files': train.iloc[:, 0], 'label': labels_train})

        test = df_label[train_size:nb_data]
        labels_test = [label] * (nb_data - train_size)
        test = pd.DataFrame(
            {'files': test.iloc[:, 0], 'label': labels_test})
        return train, test

    def create_dataframe_multi(self, label_list, nb_data):
        test_list = []
        train_list = []
        for element in label_list:
            train_name = 'train_' + element
            test_name = 'test_' + element
            train_name, test_name = self.data_preprocess_multi(element, nb_data)
            test_list.append(test_name)
            train_list.append(train_name)
        train = pd.concat(train_list, axis=0).drop_duplicates(subset=['files'])
        test = pd.concat(test_list, axis=0).drop_duplicates(subset=['files'])

        try:
            assert test['files'].nunique() == len(test)
            assert train['files'].nunique() == len(train)
            print("Assertions Passed! Sets  Are of image files W/O Duplication")
        except AssertionError:
            print("Assertions Failed")

        train = shuffle(train)
        test = shuffle(test)

        return train, test

    @staticmethod
    def generator_splitter_multi(train, test, imagepath):
        """
        function uses the `ImageDataGenerator` class
        # load our dataset as an iterator (not keeping it all in memory at once).
        :param train:
        :param test:
        :param imagepath:
        :return: data split for train val and test
        """
        # Train Set
        # tf.config.list_physical_devices()
        train['label'] = train['label'].astype(str)
        img_gen = ImageDataGenerator(validation_split=0.2)

        train_data = img_gen.flow_from_dataframe(train,
                                                 directory=imagepath,
                                                 x_col='files',
                                                 y_col='label',
                                                 featurewise_std_normalization=True,
                                                 class_mode='categorical',
                                                 batch_size=64,
                                                 target_size=(224, 224),
                                                 subset='training')

        # Validation Set
        valid_data = img_gen.flow_from_dataframe(train,
                                                 directory=imagepath,
                                                 x_col='files',
                                                 y_col='label',
                                                 featurewise_std_normalization=True,
                                                 class_mode='categorical',
                                                 batch_size=64,
                                                 target_size=(224, 224),
                                                 subset='validation')

        # Test Set
        img_gen_test = ImageDataGenerator()
        test_data = img_gen_test.flow_from_dataframe(test,
                                                     directory=imagepath,
                                                     x_col='files',
                                                     y_col='label',
                                                     featurewise_std_normalization=True,
                                                     class_mode=None,
                                                     target_size=(224, 224),
                                                     batch_size=64,
                                                     shuffle=False)
        return train_data, valid_data, test_data

