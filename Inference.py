from Classes.LoadModel import BaseModel
from Classes import Predict
import memory_profiler

from Classes.Predict import analyze_face
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


# Get img list
img_list = os.listdir(IMAGE_PATH)

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
for file in random.choices(img_list, k=3):
    result = []
    tic = time()
    file = os.path.join(IMAGE_PATH, file)
    for model_name, label in best_pairs:
        pos, neg = f'{label}: V', f'{label}: X'
        model = load_best_model(model_name, label)
        # running rage models
        result_rage = analyze_face(file)
        # running binary models
        result.append(Predict.predict_file(model, file, pos, neg)+result_rage)

    toc = time()
    run = toc - tic
    print(f'Avg Time inference {(run / 60)/len(models_list):.2f} minutes.')
    img = mpimg.imread(file)
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.text(s=result, x=190, y=100)
    plt.xticks([])
    plt.yticks([])
    plt.show()
