import pickle
import os
import numpy as np
from flask import Flask
from flask import request
from flask import make_response
from .errors import NotValidRequest
from regression.analyze import Regression, Correlation
from regression.serializers import RegressionDictSerializer
from pandas import DataFrame

app = Flask(__name__)
MODEL_SAVE_FILE = 'sources/model.sav'


@app.route('/regression/fit', methods=['POST'])
def fit():
    request_data = request.json
    validate_request(request_data)
    df = DataFrame(request_data['rows'], columns=request_data['columns'])
    correlation = Correlation(df, request_data['relevance_min_value'])
    regression = Regression(df, request_data['ignore_columns'], correlation)
    regression.calculate_metrics()
    pickle.dump(regression, open(MODEL_SAVE_FILE, 'wb'))
    serializer = RegressionDictSerializer(regression)

    return make_response({
        'error': False,
        'data': serializer.serialize()
    })


@app.route('/regression/predict', methods=['POST'])
def predict():
    if not os.path.isfile(MODEL_SAVE_FILE):
        prediction = []
    else:
        regression = pickle.load(open(MODEL_SAVE_FILE, 'rb'))
        params = regression.params_names.tolist()
        request_data = request.json
        keys = request_data.keys()
        for param in params:
            if param in keys:
                continue
            raise NotValidRequest("Missing {} param for prediction".format(param))

        prediction = regression.predict(np.array([list(request_data.values())])).tolist()

    return make_response({
        'error': False,
        'data': {
            'prediction': prediction[0]
        }
    })


@app.route('/regression/model/params')
def get_model_params():
    if not os.path.isfile(MODEL_SAVE_FILE):
        params = []
    else:
        regression = pickle.load(open(MODEL_SAVE_FILE, 'rb'))
        params = regression.params_names.tolist()

    return make_response({
        'error': False,
        'data': {
            'params': params
        }
    })


def validate_request(request_data: dict):
    keys = request_data.keys()
    all_columns = ('columns', 'ignore_columns', 'relevance_min_value', 'rows')
    list_columns = ('columns', 'ignore_columns', 'rows')
    for key in all_columns:
        if key in keys:
            continue
        raise NotValidRequest("key {} is required".format(key))

    columns_len = len(request_data['columns'])
    for column in list_columns:
        if isinstance(request_data[column], list):
            continue
        raise NotValidRequest("Value of key {} must be a list".format(column))

    for index, row in enumerate(request_data['rows']):
        row_len = len(row)
        if row_len == columns_len:
            continue
        raise NotValidRequest("Row {} length is {}, but must be {}".format(index, row_len, columns_len))


@app.errorhandler(NotValidRequest)
def error_handler(error):
    return make_response({
        "error": True,
        "data": {
            "message": error.message
        }
    }), 400
