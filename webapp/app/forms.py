from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, MultipleFileField, SelectField
from wtforms.validators import DataRequired, Required


class URLForm(FlaskForm):
    url = StringField('Image URL', validators=[DataRequired()])
    task = SelectField('Task', choices=['Classification', 'Tagging'], validators=[Required()])
    submit = SubmitField('Submit')


class FilesForm(FlaskForm):
    files = MultipleFileField('Upload images', validators=[DataRequired()])
    task = SelectField('Task', choices=['Classification', 'Tagging'], validators=[Required()])
    submit = SubmitField('Submit')
