import sklearn
from flask import Flask, request
import pickle
from Inference import *
app = Flask(__name__)

with open('model.pickle', 'rb') as p:
    model = pickle.load(p)

# http://127.0.0.1:5000/predict_single?url_image=file
# http://127.0.0.1:5000/predict_single?


@app.route('/predict_single', methods=['GET'])
def predict_single():
    # file = request.args.get('image_url')
    result = inference(file, best_pairs)
    return result

# @app.route('/predict_all/', methods=['POST'])
# def predict_all():
    # data = request.get_json()
    # df = pd.DataFrame(data)
    # df['Prediction'] = model.predict(df)
    # return df.to_json(orient='records')


if __name__ == '__main__':
    file = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_000174.jpg'
    app.run()
    # port = os.environ.get('PORT')
    # if port:
    #     app.run(host='0.0.0.0', port=int(port))
    # else:
    #     app.run()
