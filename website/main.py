from flask import Flask, render_template, flash, redirect, url_for, session, send_file
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField, RadioField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
import pandas as pd
import io
from io import BytesIO
import os
import zipfile
from contextlib import redirect_stdout
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  
Bootstrap(app)

class NameForm(FlaskForm):
    data_type = RadioField('Dataset Type', choices=[('0', 'Cross-Sectional Data'), ('1', 'Time Series Data')], validators=[DataRequired()])
    model_type = RadioField('Model Type', choices=[('0', 'Regression'), ('1', 'Classification')], validators=[DataRequired()])
    sampling_response = RadioField('Should we perform sampling if needed?', choices=[('0','Yes'),('1','No')], validators=[DataRequired()])
    visualization_need = RadioField('Do you want Visualization for your data?',choices =[('0','Yes'),('1','No')], validators=[DataRequired()])
    csv_file = FileField('Upload CSV', validators=[FileAllowed(['csv'], 'CSV files only!')])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/homepage', methods=['GET', 'POST'])
def index():
    form = NameForm()
    if form.validate_on_submit():
        if form.csv_file.data:
            session['data_type'] = form.data_type.data
            session['model_type'] = form.model_type.data
            session['sampling_response'] = form.sampling_response.data
            session['visualization_need'] = form.visualization_need.data
            
            # Read the CSV file and store its content as a string in the session
            csv_content = form.csv_file.data.read().decode('utf-8')
            session['csv_content'] = csv_content
            return redirect(url_for('result'))
    return render_template('index.html', form=form)


@app.route('/download_zip')
def download_zip():
    memory_file = BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('example.txt', 'This is the content of the text file')
        # Can add more files here
        # zf.write('/path/to/file', 'filename_in_zip')
    
    # Move to the beginning of the BytesIO object
    memory_file.seek(0)
    
    # Send the file for download
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='archive.zip'
    )

@app.route('/results', methods=['GET','POST'])
def result():
    csv_content = session.get('csv_content')
    if csv_content:
        df = pd.read_csv(io.StringIO(csv_content))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df.info()
    df_info = buffer.getvalue()

    #call preprocess function
    #data_type: imputation
    #call visulzation function
    
    return render_template('result.html', 
                           data_type=session.get('data_type'), 
                           model_type=session.get('model_type'), 
                           sampling_response=session.get('sampling_response'),
                           visualization_need = session.get('visualization_need'),
                           csv_content=df_info)

if __name__ == '__main__':
    app.run(debug=True)