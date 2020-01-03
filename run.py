import os
import sys

args = sys.argv[1:]
if len(args) != 0 and args[0] == 'serve':
    from api.application import Application
    import api.config
    app = Application(api.config)
    app.run()
else:
    from regression.application import Application
    MAIN_PATH = os.getcwd()
    SOURCES_PATH = os.path.join(MAIN_PATH, 'sources')
    app = Application(SOURCES_PATH)
    app.run()

