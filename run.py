import os
import sys

args = sys.argv[1:]
if len(args) != 0 and args[0] == 'serve':
    from api.application import app
    import api.config
    app.run(api.config.FLASK_HOST, api.config.FLASK_PORT, api.config.FLASK_DEBUG)
else:
    from regression.application import Application
    MAIN_PATH = os.getcwd()
    SOURCES_PATH = os.path.join(MAIN_PATH, 'sources')
    app = Application(SOURCES_PATH)
    app.run()

