import logging

from flask import Flask

from .config import Config
from .db import Database
from .icons import icon
from .inference import Predictor


def create_app():
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    app.config.from_object(Config)
    app.jinja_env.globals["icon"] = icon
    app.extensions["db"] = Database(app.config["DATABASE_URL"])
    app.extensions["predictor"] = Predictor(app.config["MODEL_PATH"], app.config["THRESHOLD"])
    from .routes import bp
    app.register_blueprint(bp)
    return app