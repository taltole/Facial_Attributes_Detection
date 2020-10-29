# Imports

from config import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import optimizers
from sklearn.utils import shuffle


# todo: Add option to read from libraries
# todo: Add main class to call relevant class for


class Train:
    def __init__(self, index_file, image_path):
        self.index_file = index_file
        self.image_path = image_path

    # ######################    Data Preparation   ######################
    # @staticmethod
    def img_preprocess(self, data, img_size):
        data_img = []
        IMG_WIDTH = img_size
        IMG_HEIGHT = img_size
        for i in data.iloc[:, 0]:
            if self.image_path.endswith('1'):
                file_path = find_imagepath(i)
                image = cv2.imread(file_path)
            else:
                file_path = self.image_path
                image = cv2.imread(file_path+'/'+i)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.cv2.CAP_OPENNI_GRAY_IMAGE)
            image = np.array(image).astype('float32') / 255.
            data_img.append(image)
        return data_img

    def data_preprocess(self, index_file, labels, balance, binary, img_size=224):
        """
        function read "files list.csv" and return train test filename and pixel represtation for the called label
        params: index_file - CSV for indexed labels
                labels - list of labels from csv columns name
                balance - int for a specific balanced set size
                binary - bool if user want also the negative class (= 0_  +  labels)
        returns: balanced train test set for positive and negative or multiclass labels
        """
        # read labels file list
        if isinstance(labels, str):
            df = pd.read_csv(index_file, usecols=[labels])
        else:
            df = pd.read_csv(index_file, usecols=labels)

        df_label = df[(df[labels] != 0) & (df[labels] != '0')]
        if len(df_label) < balance:
            print(f'The number of sample ({balance}) you asked for the label {labels} is higher than the number of '
                  f'sample available {len(df_label)}\nProcess Continue with {len(df_label)} Images')
            balance = len(df_label)

        # Get the label folder if not provide or label files are mix in the same folder
        #     folder = df_label[labels].apply(lambda x: '_'.join(str(x).split('_')[:-1])).unique()
        #     folder = [f for f in folder if f]

        # Train Test Split
        if balance is None:
            train_size = int(len(df_label) * 0.8)
        else:
            train_size = int(balance * 0.8)  # int(input('Please enter 2nd class train size: '))

        train = df_label[:train_size]
        test = df_label[train_size:balance]

        print(f"Starting Image Preprocessing")
        # Preprocess train image
        train_img = self.img_preprocess(train, img_size)
        class_label = [np.zeros(len(train)) if labels[0].isdigit() else np.ones(len(train))]
        train = pd.DataFrame(
            {'files': train.iloc[:, 0], 'label': np.array(*class_label).astype(str), 'image': pd.Series(train_img)})

        # Preprocess test image
        test_img = self.img_preprocess(test, img_size)
        class_label = [np.zeros(len(test)) if labels[0].isdigit() else np.ones(len(test))]
        test = pd.DataFrame({'files': test.iloc[:, 0], 'label': np.array(*class_label).astype(str), 'image': test_img})
        print('Done!')

        if binary:
            print('Creating Negative Class')
            # check for balanced data
            try:
                assert pd.read_csv(index_file, usecols=['0_' + labels]).shape[0] == balance
            except AssertionError:
                print(f"Negative class files:\t{pd.read_csv(index_file, usecols=['0_' + labels]).shape[0]}")
                print(f'Taking {balance} Images')
            # Add Negative class
            train_n, test_n = self.data_preprocess(IND_FILE, '0_' + labels, balance, binary=False)
            train = pd.concat([train, train_n], axis=0)
            test = pd.concat([test, test_n], axis=0)
            print('Shape with Negative class:')
        print(f'Train shape: \t{np.array(train).shape}\nTest shape: \t{np.array(test).shape}')

        # Verification
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
    def test_set_check(test):
        """
        Function plot images from test set for verification of good visual labels split
        :param test: test df output from data_preprocess function
        :return: plot images by label
        """
        labels = [test['files'][test['label'] == '1.0'], test['files'][test['label'] == '0.0']]
        titles = ['Positive', 'Negative']

        for l, j, t in zip(labels, range(1, 3), titles):
            plt.figure(figsize=(20, 10))
            for i in range(5):
                file = l.sample().values[0]
                file_path = find_imagepath(file)
                img = mpimg.imread(file_path)
                ax = plt.subplot(2, 5, i + 1)
                plt.suptitle(t)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
            plt.show()

    def generator_splitter(self, train, test, imagepath):
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
                                                 class_mode='binary',
                                                 batch_size=64,
                                                 target_size=(224, 224),
                                                 subset='training')

        # Validation Set
        valid_data = img_gen.flow_from_dataframe(train,
                                                 directory=imagepath,
                                                 x_col='files',
                                                 y_col='label',
                                                 class_mode='binary',
                                                 batch_size=64,
                                                 target_size=(224, 224),
                                                 subset='validation')

        # Test Set
        img_gen_test = ImageDataGenerator()
        test_data = img_gen_test.flow_from_dataframe(test,
                                                     directory=imagepath,
                                                     x_col='files',
                                                     y_col='label',
                                                     class_mode=None,
                                                     target_size=(224, 224),
                                                     batch_size=64,
                                                     shuffle=False)
        return train_data, valid_data, test_data

    def start_train(self, model, savefile, train_set, valid_set, epoch, callback=None, optimize=None, multi=None):
        """
        :param model: update base model with top layers
        :param savefile: name of the model's save file
        :param train_set:
        :param valid_set:
        :param callback = None, for using builtin callback list or providing another list of callbacks
        :param optimize = None, for using builtin optimizer or providing another compiler (loss=[], metrics=[], optimize=[])
        :param multi: if None run binary parameters, else loss=multi_class_croosentropy
        :return: history after val fitting
        """
        # todo add option for multiclass

        # Callbacks
        if callback is None:
            earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')
            checkpoint = ModelCheckpoint(savefile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            callback_list = [earlystopper, checkpoint]
        else:
            callback_list = callback

        # Optimizing
        if optimize is None:
            model.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=["accuracy"])
        else:
            model.compile(optimize)

        # Fitting
        STEP_SIZE_TRAIN = train_set.n // train_set.batch_size
        STEP_SIZE_VALID = valid_set.n // valid_set.batch_size
        history = model.fit(train_set,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_set,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=callback_list,
                            epochs=epoch)

        return history, model



