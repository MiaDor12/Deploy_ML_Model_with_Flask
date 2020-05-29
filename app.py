import numpy as np
from flask import Flask, request, render_template, url_for, jsonify, redirect, flash, get_flashed_messages
from wtforms import Form, validators, FloatField, SelectField
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

class ApplyForm(Form):
    gender = SelectField('gender', [validators.DataRequired()])
    age = FloatField('age', [validators.NumberRange(min=1, max=100, message=None), validators.DataRequired()])
    fever = FloatField('fever', [validators.NumberRange(min=35, max=43, message=None), validators.DataRequired()])
    headache = SelectField('headache', [validators.DataRequired()])
    cough = SelectField('cough', [validators.DataRequired()])
    shortness_of_breath = SelectField('shortness_of_breath', [validators.DataRequired()])

@app.route('/')
def home():
    box_2_title = "Fill your symptoms and check!"
    return render_template('index.html', box_2_title=box_2_title)

@app.route('/predict',methods=['POST'])
def predict():

    box_2_title = "Thank you for checking!"

    gender = 1 if request.form.get("gender") == 'Male' else 0
    age_60_and_above = int(float(request.form.get("age")) >= 60)
    fever = int(float(request.form.get("fever")) >= 37.5)
    headache = 1 if request.form.get("headache") == 'Yes' else 0
    cough = 1 if request.form.get("cough") else 0
    shortness_of_breath = 1 if request.form.get("shortness_of_breath") else 0
    sore_throat = 1 if request.form.get("sore_throat") else 0

    features_list = [gender, age_60_and_above, fever, headache, cough, shortness_of_breath, sore_throat]
    features_array = [np.array(features_list)]

    prediction = model.predict_proba(features_array)[:,1][0]

    if prediction < 0.55:
        output_1 = "You are perfectly healthy."
        class_type = "good"
    elif prediction <= 0.65:
        output_1 = "There is only a small chance you have Coronavirus."
        class_type = "medium"
    else:
        output_1 = "You should call the doctor."
        class_type = "bad"

    return render_template('index.html', output_1=output_1 ,class_type=class_type, box_2_title=box_2_title)


if __name__ == "__main__":
    app.run(debug=True)