from __future__ import print_function
import sys
import os
import glob
import re

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from model import get_pred, clear_uploads

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = "uploads"
        file_path = basepath + "/" + f.filename
        f.save(file_path)

        result = get_pred(file_path)
        clear_uploads(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run()
