from flask import Flask
from flask import request
from flask import make_response
from .errors import NotValidRequest

app = Flask(__name__)


@app.route('/regression/fit', methods=['POST'])
def fit():
    request_data = request.json
    validate_request(request_data)

    return make_response(request_data)


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
