import json
from flask import Flask, jsonify, request
from prediction import predict
from prediction import eci_predict

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
    response_data = {
        "response_type": "comment",
        "username": "ECI AI",
        "icon_url": "https://mattermost.com/wp-content/uploads/2022/02/icon.png",
        "text": "rag_response"
    }
    return jsonify(response_data)
    # return jsonify(predict(body)["prediction"].rpartition("Helpful Answer:")[-1])

# @application.route('/ecichat', methods=['POST'])
# def eci_prediction():
#     data = request.data or '{}'
#     body = json.loads(data)
#     print(body)
#     # rag_response = eci_predict(body)["prediction"].rpartition("Helpful Answer:")[-1]
#     response_data = {
#         "response_type": "comment",
#         "username": "ECI AI",
#         "icon_url": "https://mattermost.com/wp-content/uploads/2022/02/icon.png",
#         "text": "rag_response"
#     }
#     response_body = json.dumps(response_data)
#     return response_body
    # response_headers = [
    #     ('Content-Type', 'application/json'),
    #     ('Content-Length', str(len(response_body))),
    # ]
    # start_response('200 OK', response_headers)
    # return jsonify(predict(body)["prediction"].rpartition("Helpful Answer:")[-1])
