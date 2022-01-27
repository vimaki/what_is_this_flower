from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms.fields import SubmitField

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif']


class UploadForm(FlaskForm):
    upload_image = FileField('image', validators=[
        FileRequired(),
        FileAllowed(
            ALLOWED_EXTENSIONS,
            f'Unsupported image format. It must be one of the following: {", ".join(ALLOWED_EXTENSIONS)}'
        )
    ])
    submit = SubmitField('identify flower')
