from flask import Flask


class Application:

    def __init__(self, config):
        self.instance = Flask(__name__)
        self.config = config

    def run(self):
        self.instance.run(self.config.FLASK_HOST, self.config.FLASK_PORT, self.config.FLASK_DEBUG)
