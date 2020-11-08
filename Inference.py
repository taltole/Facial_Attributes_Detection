from Classes.LoadModel import BaseModel
from Classes import Predict
import memory_profiler
from config import *
import random

MODEL = 0
LABEL = 1


def load_best_model(model):
    best_model = model.split('/')[-1].split('_')[0]
    label = model.strip('.h5').split('_')[1:]

    # Loading Best Model
    basemodel = BaseModel(best_model)
    model = basemodel.load_model(False)
    model = basemodel.adding_toplayer(model)
    model.load_weights(model)
    print(f"\nBest Model {best_model}'s Arc. and Weights Loaded!")
    return model, label


# Get img list
img_list = os.listdir(IMAGE_PATH)

# Taking Best Model per Att
models_list = [file for file in os.listdir(MODEL_PATH) if str(file).endswith('h5')]
best_model_list = []
label_list = []
best_pairs = []
model = '/Users/tal/Dropbox/Projects/vggface_Eyeglasses.h5'

for model in models_list:
    best_model_list.append(load_best_model(model)[MODEL])
    label_list.append(load_best_model(model)[LABEL])
best_pairs.append(list(zip(best_model_list, label_list)))


print('Running Inference...')
for file in random.choices(img_list, k=3):
    result = []
    tic = time()
    file = os.path.join(IMAGE_PATH, file)
    for model, label in best_pairs:
        pos, neg = f'{label}: V', f'{label}: X'
        result.append(Predict.predict_file(model, file, pos, neg))

    toc = time()
    run = toc - tic
    print(f'Time inference {(run / 60):.2f} minutes.')
    img = mpimg.imread(file)
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.text(s=result, x=190, y=100)
    plt.xticks([])
    plt.yticks([])
    plt.show()
