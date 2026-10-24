import logging

from flask import Flask

from .config import Config
from .db import Database
from .icons import icon
#from .inference import Predictor


def create_app():
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    app.config.from_object(Config)
    app.jinja_env.globals["icon"] = icon
    app.extensions["db"] = Database(app.config["DATABASE_URL"])
    if app.config["INFERENCE_MODE"] == "remote":
        from .remote import RemotePredictor
        app.extensions["predictor"] = RemotePredictor(
            app.config["API_URL"], app.config["API_KEY"], app.config["API_TIMEOUT"])
        logging.getLogger("visionguard").info("Inference mode: REMOTE -> %s", app.config["API_URL"])
    else:
        from .inference import Predictor
        app.extensions["predictor"] = Predictor(app.config["MODEL_PATH"], app.config["THRESHOLD"])
        logging.getLogger("visionguard").info("Inference mode: LOCAL (%s)", app.config["MODEL_PATH"])
    from .routes import bp
    app.register_blueprint(bp)
    return app