import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, url_for

import model_inference
from forms import UploadForm

UPLOAD_FOLDER = 'static/img/'

# Loading environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    logging.warning('Invalid environment variables!')
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')


def change_image_name(filename):
    if filename.find('.') != -1:
        _, extension = filename.rsplit('.', 1)
        new_filename = f'uploaded_image.{extension.lower()}'
        return new_filename
    else:
        raise ValueError('No extension in this filename')


@app.route('/', methods=['GET', 'POST'])
def predict():
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
    return redirect(url_for('static', filename='img/' + filename), code=301)


if __name__ == '__main__':
    app.run()

# import model_inference
#
# image_path = 'flower.jpeg'
#
# with open(image_path, 'rb') as img:
#     byte_im = img.read()
#
# res = model_inference.get_inference(byte_im)
# print(res)
