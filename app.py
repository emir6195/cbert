from flask import Flask
from routes.cbert import route
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30720"


app = Flask(__name__)
app.register_blueprint(route, url_prefix='/cbert')


if __name__ == '__main__':
    app.run("0.0.0.0", 5003)