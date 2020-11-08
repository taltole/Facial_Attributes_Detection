from tensorflow.python.keras.optimizers import RMSprop

from Classes.LoadModel import BaseModel
from Classes.Predict import Prediction
import memory_profiler

from config import *

# todo load all H5 / CLS
# todo check and add best model _att combination
# todo getting img path to iterate over


tic = time()

# Get img path
IMG_PATH = IMAGE_PATH
img_list = os.listdir(IMG_PATH)

# Best Model per att
# find best combination
# # models_list = os.listdir(MODEL_PATH)

MODELS = '/Users/tal/Dropbox/Projects/vggface_Eyeglasses.h5'
best_model = MODELS.split('/')[-1].split('_')[0]
label = MODELS.strip('.h5').split('_')[-1]

# Loading Best Model
# history = json.load(open(PATH_JSON))

basemodel = BaseModel(best_model)
model = basemodel.load_model(False)
model = basemodel.adding_toplayer(model)
model.load_weights(MODELS)
# model.compile(RMSprop(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=["accuracy"])
print(f"\nBest Model {best_model}'s Arc. and Weights Loaded!")

# Evaluate the network on valid data
# Prediction.evaluate_model(model, valid_data)

# Predict on test data
# y_pred = Prediction.test_prediction(model, test_data, train_data)
# plot
# top = min(len(test['label']), len(y_pred))
# metrics = Metrics(history, epoch, test['label'][:top].tolist(), y_pred[:top], model_name, label)
# metrics.confusion_matrix()
# #metrics.acc_loss_graph()
# metrics.classification_report()

# Inference
# labels = [test['files'][test['label'] == '1.0'], test['files'][test['label'] == '0.0']]
# '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/3/face_att_174563.jpg'

pos, neg = f'With {label}', f'W/O {label}'
for file in img_list[15:45]:
    file = os.path.join(IMG_PATH, file)
    Prediction.predict_file(model, file, pos, neg)
    Prediction.analyze_face(IND_FILE)
