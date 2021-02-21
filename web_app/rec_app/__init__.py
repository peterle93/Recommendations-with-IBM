from flask import Flask

app = Flask(__name__)

# The following command invoke run.property from rec_app directory.
from rec_app import run
