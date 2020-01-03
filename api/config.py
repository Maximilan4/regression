import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), '.env'))

FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
FLASK_PORT = os.getenv('FLASK_PORT', '5000')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', False)
