from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, MultipleFileField
from wtforms.validators import DataRequired


class URLForm(FlaskForm):
    url = StringField('Image URL', validators=[DataRequired()])
    submit = SubmitField('Classify')


class FilesForm(FlaskForm):
    files = MultipleFileField('Upload images', validators=[DataRequired()])
    submit = SubmitField('Classify')
