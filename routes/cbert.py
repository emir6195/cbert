from flask import Blueprint, request, jsonify, abort
from lib.CBERT import cbert
route = Blueprint('cbert', __name__)


@route.route('/train', methods=["POST"])
def train():
    data = request.get_json()
    if 'docs' in data and 'index_name' in data:
        docs = data['docs']
        index_name = data['index_name']
        success = cbert.train(docs, index_name)
        return jsonify({"success": success})
    else:
        return abort(400)
    
@route.route('/query', methods=["POST"])
def query():
    data = request.get_json()
    if 'query' in data and 'index_name' in data:
        query = data['query']
        index_name = data['index_name']
        results = cbert.invoke(index_name, query)
        return jsonify({"results": results})
    else:
        return abort(400)