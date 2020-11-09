from Classes.LoadModel import BaseModel
from Classes import Predict
import memory_profiler

from Classes.Predict import analyze_face, Prediction
from sklearn.metrics import accuracy_score
from config import *
import random

MODEL = 0
LABEL = 1


def load_best_model(model_name, label):
    # Loading BaseModel
    basemodel = BaseModel(model_name)
    model = basemodel.load_model(False)
    model = basemodel.adding_toplayer(model)
    # Loading Best Model
    model.load_weights(os.path.join(MODEL_PATH, f'{model_name}_{label}.h5'))
    print(f"\nBest Model {model_name}'s Arc. and Weights Loaded!")
    return model


# Taking Best Model per Att
models_list = [file for file in os.listdir(MODEL_PATH) if str(file).endswith('h5')]
best_model_list = []
label_list = []
best_pairs = []
# modelz = '/Users/tal/Dropbox/Projects/vggface_Eyeglasses.h5'

for model in models_list:
    best_model = model.split('/')[-1].split('_')[MODEL]
    label = ''.join(model.strip('.h5').split('_')[LABEL:])
    best_model_list.append(best_model)
    label_list.append(label)

best_pairs = zip(best_model_list, label_list)

print('Running Inference...')
label_query = ['Eyeglasses']


def inference(image_path, best_pairs, plot=False):

    # Get img list
    img_list = os.listdir(IMAGE_PATH)
    file = ''.join(random.choices(img_list, k=1))
    result = []
    score = []
    tic = time()
    file = os.path.join(image_path, file)

    for model_name, label in best_pairs:
        pos, neg = f'{label}: V', f'{label}: X'
        model = load_best_model(model_name, label)

        # running rage models
        result_rage, score_rage = analyze_face(file)
        # running binaryCls models
        bicls_result, bicls_score = Predict.predict_file(model, file, pos, neg)
        # running multiCls models ** I changed only here the label dict onmain it stays the same
        # labels_hair = {0: 'Bald', 1: 'Black_Hair', 2: 'Blond_Hair', 3: 'Brown_Hair', 4: 'Gray_Hair'}
        # result.append(Prediction.predict_label_multi(model, labels_hair, file, 'ResNet50'))

        result.append(bicls_result+result_rage)
        file_dict = {file: {label: score}}

        # update dictionary
        json_file = PATH_JSON + 'sum_file.json'
        with open(json_file, 'r') as jf:
            json_obj = json.load(jf)
        json_obj[file] = file_dict
        json_obj[file].update(score_rage)
        json_obj[file][label] = bicls_score[0][0].astype(float)

        with open(json_file, "w") as jf_out:
            json.dump(json_obj, jf_out)

    toc = time()
    run = toc - tic
    print(f'Avg Time inference {(run / 60)/len(models_list):.2f} minutes.')

    if plot:
        result = ''.join(result)
        img = mpimg.imread(file)
        plt.figure(figsize=(8, 5))
        plt.imshow(img)
        plt.text(s=result, x=190, y=100)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return file_dict


result = inference(IMAGE_PATH, best_pairs)
print(result)

# sum_json =
# print(sum_json)