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
    name = model_name
    basemodel = BaseModel(model_name[:-1])
    model = basemodel.load_model(False)
    model = basemodel.adding_toplayer(model, name)
    # Loading Best Model
    model.load_weights(os.path.join(MOD_ATT_PATH, f'{model_name}_{label}.h5'))
    print(f"\nBest Model {model_name}'s Arc. and Weights Loaded!")
    return model


# Taking Best Model per Att
models_list = [file for file in os.listdir(MOD_ATT_PATH) if str(file).endswith('h5')]
best_model_list = []
label_list = []
best_pairs = []

for model in models_list:
    best_model = model.split('/')[-1].split('_')[MODEL]
    label = model.strip('.h5').split('_', 1)[LABEL:]
    label = ''.join(label)
    best_model_list.append(best_model)
    label_list.append(label)

best_pairs = zip(best_model_list, label_list)

print('Running Inference...')


def inference(file, best_pairs, plot=True):
    scores = []
    tic = time()

    # running rage models
    result_rage, score_rage = analyze_face(file)

    result = [result_rage]
    labels, scores = [], []
    lbl_scr_dict = dict()
    file_dict = {file: lbl_scr_dict}

    for model_name, label in best_pairs:
        pos, neg = f'{label}: V', f'{label}: X'
        model = load_best_model(model_name, label)

        # running multiCls models
        if label == 'Hair_color':
            labels_hair = {0: 'Bald', 1: 'Black_Hair', 2: 'Blond_Hair', 3: 'Brown_Hair', 4: 'Gray_Hair'}
            result_mult, score_multi = Prediction.predict_label_multi(model, labels_hair, file, 'ResNet50')
            color = result_mult
            result.append(color+'\n')
            scores.append(score_multi.astype(float))
            labels.append(label)

        else:
            # running binaryCls models
            result_bicls, score_bicls = Predict.predict_file(model, file, pos, neg)
            result.append(result_bicls + '\n')
            scores.append(score_bicls.astype(float))
            labels.append(label)

    # return dict labels score
    lbl_scr_dict = {k: v for k, v in zip(labels, scores)}
    file_dict[file] = lbl_scr_dict
    file_dict[file].update(score_rage)

    toc = time()
    run = toc - tic
    print(f'Total Run Time inference:\t {(run / 60):.2f} minutes.')

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


def main():
    # Get img list
    img_list = os.listdir(IMAGE_PATH)
    image = ''.join(random.choices(img_list, k=1))
    file_path = os.path.join(IMAGE_PATH, image)

    # Run Inference
    result = inference(file_path, best_pairs)
    print([(k, v) for k, v in result.items()], sep='\n')
    return result


if __name__ == '__main__':
    main()

