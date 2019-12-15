import os
from regression.application import Application

MAIN_PATH = os.getcwd()
SOURCES_PATH = os.path.join(MAIN_PATH, 'sources')

app = Application(SOURCES_PATH)
app.run()
