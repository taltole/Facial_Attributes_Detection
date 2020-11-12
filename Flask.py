import sklearn
from flask import Flask, request, render_template
import pickle
from Inference import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_single', methods=['GET'])
def predict_single():
    models_list = [file for file in os.listdir(MOD_ATT_PATH) if str(file).endswith('h5')]
    best_model_list = []
    label_list = []

    for model in models_list:
        best_model = model.split('/')[-1].split('_')[MODEL]
        label = model.strip('.h5').split('_', 1)[LABEL:]
        label = ''.join(label)
        best_model_list.append(best_model)
        label_list.append(label)

    best_pairs = zip(best_model_list, label_list)
    file = request.args.get('image_url')
    result = inference(file, best_pairs)
    return result


if __name__ == '__main__':
    app.run()
