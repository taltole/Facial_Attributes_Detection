import sklearn
from flask import Flask, request
import pickle
from Inference import *
app = Flask(__name__)

# with open('model.pickle', 'rb') as p:
#     model = pickle.load(p)

# http://127.0.0.1:5000/predict_all?url_image=file
# http://127.0.0.1:5000/predict_single?
# http://127.0.0.1:5000/predict_single?image_url=/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_029252.jpg


@app.route('/predict_single', methods=['GET'])
def predict_single():
    file = request.args.get('image_url')
    result = inference(file, best_pairs)
    return result


@app.route('/predict_all/', methods=['POST'])
def predict_all():

    data = request.get_json('image_url')
    result = inference(data, best_pairs)
    #
    # df = pd.DataFrame(data)
    # df['Prediction'] = model.predict(df)

    return result  # df.to_json(orient='records')


if __name__ == '__main__':
    # img_list = os.listdir(IMAGE_PATH)
    # image = ''.join(random.choices(img_list, k=1))
    # file = os.path.join(IMAGE_PATH, image)
    # print(file)
    file = '/Users/tal/Google Drive/Cellebrite/Datasets/face_att/1/face_att_029252.jpg'
    app.run()
    port = os.environ.get('PORT')
    # if port:
    #     app.run(host='0.0.0.0', port=int(port))
    #     print('in port')
    # else:
    #     app.run()