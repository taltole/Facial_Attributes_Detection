from deepface import DeepFace

from config import *
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_MNV2
from tensorflow.keras.applications.resnet50 import preprocess_input as Preprocess_RESNET50


class Prediction:

    @staticmethod
    def evaluate_model(model, valid_data):
        """
        :param model: model
        :param valid_data: validation data set
        Evalutates the network on the validation set
        """
        print('Evaluating the network ...')
        loss, acc = model.evaluate(valid_data)
        print(f"Validation Loss:\t{round(loss, 3)}\nValidation Acc.:\t{round(acc, 3)}")

    @staticmethod
    def test_prediction(model, test_data, train_data):
        """
        :param model: model
        :param test_data: test data set
        :param train_data: train data set
        return: predictions on the test set
        """
        test_data.reset()
        STEP_SIZE_TEST = test_data.n // test_data.batch_size
        print('Starting prediction...')
        pred = model.predict(test_data, steps=STEP_SIZE_TEST)
        print('Done!')
        labels = train_data.class_indices
        labels = dict((v, k) for k, v in labels.items())
        if len(list(labels.keys())) > 2:
            predicted_class_indices = np.argmax(pred, axis=1)
        else:
            predicted_class_indices = list(map(lambda x: 1 if float(x) >= 0.5 else 0, pred))
        y_pred = [labels[k] for k in predicted_class_indices]
        return y_pred

# ----------------------------------------------------------------------------------
    # Inference
    @staticmethod
    def predict_label(model, labels, pos, neg):

        for label in labels:
            file = label.sample().values[0]
            imagepath = find_imagepath(file)
            img = tf.keras.preprocessing.image.load_img(os.path.join(imagepath), target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array)
            score = predictions[0]
            result = max((pos, 100 * score), (neg, 100 * (1 - score)), key=lambda x: x[1])
            text = f"{neg}:\t{100 * (1 - score)}%\t{pos}:\t{100 * score}%"
            imge = mpimg.imread(imagepath)
            plt.figure(figsize=(5, 5))
            plt.imshow(imge)
            plt.title(result[0])
            plt.xticks([])
            plt.yticks([])
        plt.show()
        return result

    @staticmethod
    def predict_label_multi(model, labels, imagepath, preprocess=None):

        img = tf.keras.preprocessing.image.load_img(os.path.join(imagepath), target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0)  # Create batch axis

        if preprocess == 'vgg16':
            img_array = preprocess_input_VGG16(img_array)
        elif preprocess == 'vgg19':
            img_array = preprocess_input_VGG19(img_array)
        elif preprocess == 'MobileNetV2':
            img_array = preprocess_input_MNV2(img_array)
        elif preprocess == 'ResNet50':
            img_array = Preprocess_RESNET50(img_array)

        predictions = model.predict(img_array)
        index = np.argmax(predictions, axis=1)
        score = predictions[int(index)]
        result = list(labels.keys())[int(score)]
        # imge = mpimg.imread(imagepath)
        # plt.figure(figsize=(5, 5))
        # plt.imshow(imge)
        # plt.title(list(labels.keys())[int(score)])
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        return result, score


def predict_file(model, file, pos, neg):
    """
    function read an image file and predict its label using model and class name arguments
    """
    # running binary models
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    result = max((pos, 100 * score), (neg, 100 * (1 - score)), key=lambda x: x[1])
    text = f"{neg}:\t{100 * (1 - score)}%\t{pos}:\t{100 * score}%"
    result = result[0]
    return result, predictions


def analyze_face(df, backend=0, plot=False):
    """
    Function call image file as str or from dataframe and analyze it with deepface module to extract
    race, age, gender and emotion
    """
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

    # reading file
    # if isinstance(df, str):
    #     file = find_imagepath(df)
    # elif isinstance(df, list):
    #     file = df
    # else:
    #     img_f1 = df.sample().values[0]
    #     file = find_imagepath(img_f1)
    file = df
    # Run DeepFace
    try:
        demography, score = DeepFace.analyze(file, detector_backend=backends[backend])
        age = int(demography['age'])
        gender = demography['gender']
        emotion = demography['dominant_emotion']
        race = demography['dominant_race']
        textstr = f'\nAge: {age}\nGender: {gender}\nRace: {race.title()}\nEmotion: {emotion.title()}\n'
        # Plot
        if plot:
            plt.figure(figsize=(5, 5))
            img = mpimg.imread(file)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        return textstr, score

    except ValueError:
        print('Face could not be detected')
        return ''



# # ### Predicting
#
#
# test_data.reset()
# pred = model.predict(test_data,
#                      steps=STEP_SIZE_TEST,
#                      verbose=1)
#
# # In[ ]:
#
#
# y_pred = list(map(lambda x: 1 if float(x) >= 0.2 else 0, pred))  # {1.0:'With',0.0:'W/O'}
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# predicted_class_indices = np.argmax(pred, axis=1)
# labels = (train_data.class_indices)
# labels = dict((v, k) for k, v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]
#
# # In[ ]:
#
#
# len(predictions)
#
# # In[ ]:
#
#
# # In[ ]:
#
#
# filenames = test_data.filenames
# s = len(predictions)
# results = pd.DataFrame({"Filename": filenames,
#                         "Predictions": predictions})
# results['Predictions'].unique()
#
#
# # In[251]:
#
#
# def analyze(img_path, actions=[], models={}, enforce_detection=True, detector_backend='opencv'):
#     if type(img_path) == list:
#         img_paths = img_path.copy()
#         bulkProcess = True
#     else:
#         img_paths = [img_path]
#         bulkProcess = False
#
#     # ---------------------------------
#
#     # if a specific target is not passed, then find them all
#     if len(actions) == 0:
#         actions = ['emotion', 'age', 'gender', 'race', 'eyeglass']
#
#     # print("Actions to do: ", actions)
#
#     # ---------------------------------
#
#     if 'emotion' in actions:
#         if 'emotion' in models:
#             print("already built emotion model is passed")
#             emotion_model = models['emotion']
#         else:
#             emotion_model = Emotion.loadModel()
#
#     if 'age' in actions:
#         if 'age' in models:
#             print("already built age model is passed")
#             age_model = models['age']
#         else:
#             age_model = Age.loadModel()
#
#     if 'gender' in actions:
#         if 'gender' in models:
#             print("already built gender model is passed")
#             gender_model = models['gender']
#         else:
#             gender_model = Gender.loadModel()
#
#     if 'race' in actions:
#         if 'race' in models:
#             print("already built race model is passed")
#             race_model = models['race']
#         else:
#             race_model = Race.loadModel()
#
#     if 'eyeglass' in actions:
#         if 'eyeglass' in models:
#             print("already built race model is passed")
#             eyeglass_model = models['eyeglass']
#         else:
#             eyeglass_model = eyeglasses_model.loads('eyeglass.h5')
#     # ---------------------------------
#
#     resp_objects = []
#
#     disable_option = False if len(img_paths) > 1 else True
#
#     global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable=disable_option)
#
#     # for img_path in img_paths:
#     for j in global_pbar:
#         img_path = img_paths[j]
#
#         resp_obj = "{"
#
#         disable_option = False if len(actions) > 1 else True
#
#         pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable=disable_option)
#
#         action_idx = 0
#         img_224 = None  # Set to prevent re-detection
#         # for action in actions:
#         for index in pbar:
#             action = actions[index]
#             pbar.set_description("Action: %s" % (action))
#
#             if action_idx > 0:
#                 resp_obj += ", "
#
#             if action == 'emotion':
#                 emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
#                 img = functions.preprocess_face(img=img_path, target_size=(48, 48), grayscale=True,
#                                                 enforce_detection=enforce_detection, detector_backend=detector_backend)
#
#                 emotion_predictions = emotion_model.predict(img)[0, :]
#
#                 sum_of_predictions = emotion_predictions.sum()
#
#                 emotion_obj = "\"emotion\": {"
#                 for i in range(0, len(emotion_labels)):
#                     emotion_label = emotion_labels[i]
#                     emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
#
#                     if i > 0: emotion_obj += ", "
#
#                     emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)
#
#                 emotion_obj += "}"
#
#                 emotion_obj += ", \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])
#
#                 resp_obj += emotion_obj
#
#             elif action == 'age':
#                 if img_224 is None:
#                     img_224 = functions.preprocess_face(img_path, target_size=(224, 224), grayscale=False,
#                                                         enforce_detection=enforce_detection)  # just emotion model expects grayscale images
#                 # print("age prediction")
#                 age_predictions = age_model.predict(img_224)[0, :]
#                 apparent_age = Age.findApparentAge(age_predictions)
#
#                 resp_obj += "\"age\": %s" % (apparent_age)
#
#             elif action == 'gender':
#                 if img_224 is None:
#                     img_224 = functions.preprocess_face(img=img_path, target_size=(224, 224), grayscale=False,
#                                                         enforce_detection=enforce_detection,
#                                                         detector_backend=detector_backend)  # just emotion model expects grayscale images
#                 # print("gender prediction")
#
#                 gender_prediction = gender_model.predict(img_224)[0, :]
#
#                 if np.argmax(gender_prediction) == 0:
#                     gender = "Woman"
#                 elif np.argmax(gender_prediction) == 1:
#                     gender = "Man"
#
#             elif action == 'eyeglass':
#                 if img_224 is None:
#                     img_224 = functions.preprocess_face(img=img_path, target_size=(224, 224), grayscale=False,
#                                                         enforce_detection=enforce_detection,
#                                                         detector_backend=detector_backend)  # just emotion model expects grayscale images
#                 # print("gender prediction")
#
#                 eyeglass_prediction = eyeglass_model.predict(img_224)[0, :]
#
#                 if np.argmax(eyeglass_prediction) == 0:
#                     eg = "W/O Eyeglasses"
#                 elif np.argmax(eyeglass_prediction) == 1:
#                     eg = "With Eyeglasses"
#
#                 resp_obj += "\"eyeglass\": \"%s\"" % (eg)
#
#             elif action == 'race':
#                 if img_224 is None:
#                     img_224 = functions.preprocess_face(img=img_path, target_size=(224, 224), grayscale=False,
#                                                         enforce_detection=enforce_detection,
#                                                         detector_backend=detector_backend)  # just emotion model expects grayscale images
#                 race_predictions = race_model.predict(img_224)[0, :]
#                 race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
#
#                 sum_of_predictions = race_predictions.sum()
#
#                 race_obj = "\"race\": {"
#                 for i in range(0, len(race_labels)):
#                     race_label = race_labels[i]
#                     race_prediction = 100 * race_predictions[i] / sum_of_predictions
#
#                     if i > 0: race_obj += ", "
#
#                     race_obj += "\"%s\": %s" % (race_label, race_prediction)
#
#                 race_obj += "}"
#                 race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])
#
#                 resp_obj += race_obj
#
#             action_idx = action_idx + 1
#
#         resp_obj += "}"
#
#         resp_obj = json.loads(resp_obj)
#
#         if bulkProcess == True:
#             resp_objects.append(resp_obj)
#         else:
#             return resp_obj
#
#     if bulkProcess == True:
#         resp_obj = "{"
#
#         for i in range(0, len(resp_objects)):
#             resp_item = json.dumps(resp_objects[i])
#
#             if i > 0:
#                 resp_obj += ", "
#
#             resp_obj += "\"instance_" + str(i + 1) + "\": " + resp_item
#         resp_obj += "}"
#         resp_obj = json.loads(resp_obj)
#         return resp_obj
#
#
# # In[259]:
#
#
# def analyze_face(df, backend=0):
#     """
#     Function call image file as str or from dataframe and analyze it with deepface module to extract
#     race, age, gender and emotion
#     """
#
#     # reading file
#     if isinstance(df, str):
#         file = find_imagepath(df)
#     else:
#         img_f1 = df.sample().values[0]
#         file = find_imagepath(img_f1)
#
#     # Run DeepFace
#     try:
#         backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
#         demography = DeepFace.analyze(file, detector_backend=backends[backend])
#         age = int(demography['age'])
#         gender = demography['gender']
#         emotion = demography['dominant_emotion']
#         race = demography['dominant_race']
#         textstr = f'Age:\t\t{age}\nGender:\t\t{gender}\nRace:\t\t{race.title()}\nEmotion:\t{emotion.title()}'
#
#     except ValueError:
#         print('Face could not be detected')
#         sys.exit()
#
#     # Plot
#     plt.figure(figsize=(5, 5))
#     img = mpimg.imread(file)
#     plt.imshow(img)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     print(textstr)
#
#
# # DeepFace.verify("img1.jpg", "img2.jpg", model_name = "VGG-Face", model = model)
#
#
# # eyeglass_model = tf.saved_model.load(os.path.join(os.getcwd(), 'eyeglass.h5'))
# model.load('eyeglass.h5')
#
# # In[261]:
#
#
# # padded_shapes = ([90000], ())
# analyze_face(results['Filename'])
#
# # In[238]:
#
#
# # DeepFace.stream(IMAGEPATH)
#
#
# # In[ ]:
