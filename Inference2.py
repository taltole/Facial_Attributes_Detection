import pickle

from Classes.LoadModel import BaseModel
from Classes import Predict
import memory_profiler
from Classes.Predict import analyze_face, Prediction
from sklearn.metrics import accuracy_score
from config import *
import random
MODEL = 0
LABEL = 1
import psutil
# gives a single float value
file = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_015191.jpg'

def run_benchmark():
    print(psutil.cpu_percent())# gives an object with many fields
    print(psutil.virtual_memory())# you can convert that object to a dictionary
    # print(dict(psutil.virtual_memory()._asdict()))# you can have the percentage of used RAM
    print(psutil.virtual_memory().percent)# you can calculate percentage of available memory
    print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)


def load_best_model(model_name):
    """
    function loads basemodel without toplayer and add customized top layers depends on model-name
    return model arch.
    """
    # Loading BaseModel
    name = model_name
    basemodel = BaseModel(model_name[:-1])
    model = basemodel.load_model(False)
    model = basemodel.adding_toplayer(model, name)
    return model

p = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/pkl'
# Taking Best Model per Att
models_list = [file for file in os.listdir(p) if str(file).endswith('SVC.pkl')]
best_model_list = []
label_list = []
best_pairs = []

for model in models_list:
    best_model = "SVC"
    label = model.strip('_SVC.pkl')
    label = ''.join(label)
    best_model_list.append(best_model)
    label_list.append(label)

best_pairs = zip(best_model_list, label_list)

print('Running Inference...')
for model in models_list:
    with open(p+'/'+model, 'rb') as p:
        model = pickle.load(p)
        result, _ = Predict.predict_file(model, file, pos='+', neg='-')


def inference(file, best_pairs, plot=True):
    """
    function take a file and list of model - att combination and run inference on it. print a labeled pic and dict.
    params: file: str with abs file path
            best pairs: list of tuples
            plot: bool if want to see labeled image file
    returns: plot and dictionary with file and labels
    """
    tic = time()
    # running rage models
    # Loading Models
    # model_bi = load_best_model('vggface1')
    # model_multi = load_best_model('ResNet507')

    result_rage, _ = analyze_face(file)
    rage_list = [(k.split(':')[0], k.split(':')[1]) for k in result_rage.split('\n')[1:-1]]
    results_img = [result_rage]
    labels = []
    lbl_scr_dict = dict()
    file_dict = {file: lbl_scr_dict}

    for model_name, label in best_pairs:
        pos, neg = f'{label}: V', f'{label}: X'
        print(f"\nFinding best model for {label}...")

        # running multiCls models
        if label == 'Hair_color':
            # Loading Multicls Model Weights
            # model_multi.load_weights(os.path.join(MOD_ATT_PATH, f'{model_name}_{label}.h5'))
            print(f"\nBest Model {model_name}'s Arc. and Weights Loaded!")

            labels_hair = {0: 'Bald', 1: 'Black_Hair', 2: 'Blond_Hair', 3: 'Brown_Hair', 4: 'Gray_Hair'}
            # result, _ = Prediction.predict_label_multi(model_multi, labels_hair, file, 'ResNet50')
            result = (label, result)
            result_img = result[1]
        # running binaryCls models
        else:
            # Loading Binary Model Weights
            model_bi.load_weights(os.path.join(MOD_ATT_PATH, f'{model_name}_{label}.h5'))
            print(f"\nBest Model {model_name}'s Arc. and Weights Loaded!")
            result, _ = Predict.predict_file(model_bi, file, pos, neg)

            if result.split(': ')[1] == 'X':
                result = ('', '')
                result_img = ''
            else:
                result_img = result.split(': ')[0]
                result = (result.split(': ')[0], '')

        results_img.append(result_img + '\n')
        rage_list.append(result)
        file_dict[file] = {k: v for k, v in rage_list}

        labels.append(label)
    file_dict[file].pop('')

    # Checking Runtime
    toc = time()
    run = toc - tic
    print(f'Total Run Time inference:\t {(run / 60):.2f} minutes.')

    if plot:
        result = ''.join(results_img)
        img = mpimg.imread(file)
        plt.figure(figsize=(8, 5))
        plt.imshow(img)
        plt.text(s=result, x=190, y=100)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return file_dict


def main():
    # Get img list
    img_list = os.listdir(IMAGE_PATH)
    image = ''.join(random.choices(img_list, k=1))
    file_path = os.path.join(IMAGE_PATH, image)

    # Run Inference
    # result = inference(file_path, best_pairs)
    # print(result)
    # return result


if __name__ == '__main__':
    main()
    # run_benchmark()
# '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_015191.jpg'
# '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_057829.jpg'
# '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_054661.jpg

