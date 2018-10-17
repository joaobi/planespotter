# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:35:00 2018

@author: joaobi
"""
#!flask/bin/python
import sys
import os
#sys.path.append("c:/projects/airlinesdetection/libs")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../libs')))
import json
import requests

# from planespotter import planespotter

from flask import Flask, request, url_for, send_from_directory, render_template
from werkzeug import secure_filename

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def show_image_html(original_img,output_img,json_pred):
       return '''
    <!doctype html>
    <title>Prediction Output</title>
    <h1>'''+output_img+'''</h1>
    <img src="'''+original_img+'''" alt="'''+original_img+'''" height="600" width="600">
    <img src="'''+output_img+'''" alt="'''+output_img+'''" height="600" width="600">
    '''+json_pred+'''
    ''' 

def show_image_html(json_pred):
       return '''
    <!doctype html>
    <title>Prediction Output</title>
    '''+json_pred+'''
    ''' 

# Create a URL route in our application for "/"

@app.route('/', methods=['POST'])
def post():
    print('***************************')
    file = request.files['file']
    print(file)
    print('***************************')

    file = request.files['file']
    if file and allowed_file(file.filename):
        print('**found file', file.filename)
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # response = requests.get('http://127.0.0.1:5000/api/prediction')
        # geodata = response.json()

        # print(geodata)

        url = 'http://127.0.0.1:5000/api/prediction'
        # data =  json.dumps({'filename' : 'EK-1778749.jpg'})
        data =  json.dumps({'filename' : filename})
        headers={'content-type':'application/json', 'accept':'application/json'}

        print("URL      >> "+url)
        print("FILENAME >> "+filename)
        print("DATA     >> "+data)
        # response = requests.post('http://127.0.0.1:5000/api/prediction', data = {'filename':filename})
        response = requests.post(url,data=data,headers=headers)
        print("REQUESTED A POST!")
        print(response.status_code)
        result = response.json()

        print(response.json())


    return render_template('main1.html', pred = result)

        # return show_image_html(image_dest,image_scored,json.dumps(json_pred))    

    # return render_template('about.html')

@app.route('/', methods=['GET'])
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """
    return render_template('main1.html', pred = {})

@app.route('/showAbout')
def showSignUp():
    return render_template('about.html')

@app.route('/showPredict')
def showPredict():

    response = requests.get('http://127.0.0.1:5000/api/prediction')
    data = response.json()

    print(data)

        # return show_image_html(json.dumps(geodata))

    return render_template('predict.html',posts = data)

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)