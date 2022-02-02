"""Initialization of the WSGI application."""

import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask

logging.basicConfig(level=logging.INFO)

# Loading environment variables
try:
    FLASK_SECRET_KEY = os.environ['FLASK_SECRET_KEY']
except KeyError:
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    else:
        logging.warning('Invalid environment variables!')
        sys.exit(1)

# Create an instance of the WSGI application.
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY

from .forms import *
from .views import *
from .model_inference import *
