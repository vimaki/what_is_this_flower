"""Module with forms used in the application.

Classes
-------
UploadForm
    Form for sending an image to the server.
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms.fields import SubmitField

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']


class UploadForm(FlaskForm):
    """Form for sending an image to the server.

    The form consists of a file upload field and a submit button.
    An additional check is performed to ensure that the file extension
    matches the allowed list.
    """

    upload_image = FileField('image', validators=[
        FileRequired(),
        FileAllowed(
            ALLOWED_EXTENSIONS,
            f'Unsupported image format. It must be one of the following: {", ".join(ALLOWED_EXTENSIONS)}'
        )
    ])
    submit = SubmitField('identify flower')
