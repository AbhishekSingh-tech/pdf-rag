import json
from flask import Flask, jsonify, request
from prediction import predict

application = Flask(__name__)


@application.route('/')
@application.route('/status')
def status():
    return jsonify({'status': 'ok'})


@application.route('/predictions', methods=['POST'])
def create_prediction():
    data = request.data or '{}'
    body = json.loads(data)
    print(body)
    return jsonify(predict(body)["prediction"].rpartition("Helpful Answer:")[-1])
