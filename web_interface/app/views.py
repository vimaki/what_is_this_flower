"""Module with routes used in the application.

Functions
---------
predict
    Setting the main content of the application.
display_image
    Display an image on the same page when a form is submitted.

References
----------
__init__.py
    A module that contains an object that is WSGI application.
model_inference.py
    A module that performing flower classification on the transferred
    image.
forms.py
    A module that contains forms used in the application.
utils.py
    A module that contains auxiliary functions for implementing the
    business logic of the application.
"""

import logging
import os

from flask import flash, redirect, render_template, url_for

from . import app
from . import model_inference
from .forms import UploadForm
from .utils import change_image_name

UPLOAD_FOLDER = 'app/static/img/'

logging.basicConfig(level=logging.INFO)


@app.route('/', methods=['GET', 'POST'])
def predict():
    """Setting the main content of the application."""
    form = UploadForm()
    if form.validate_on_submit():
        image = form.upload_image.data
        filename = change_image_name(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        flash('You can upload the next photo', category='success')
        try:
            flower_name_eng, flower_name_rus = model_inference.get_inference(image_path)
            result = (flower_name_eng, flower_name_rus)
        except:
            flash('Sorry. Try uploading a different photo', category='error')
            result = ('Oops...', 'Something went wrong in the calculation')
            logging.warning('An error occurred during an attempt to classify an image')
        return render_template('index.html', filename=filename, result=result, form=form)
    return render_template('index.html', form=form)


@app.route('/display_image/<string:filename>')
def display_image(filename):
    """Display an image on the same page when a form is submitted."""
    return redirect(url_for('static', filename='img/' + filename), code=301)
