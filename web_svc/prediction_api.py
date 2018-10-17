# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:35:00 2018

@author: joaobi
"""
#!flask/bin/python
import sys
import os
#sys.path.append("c:/projects/airlinesdetection/libs")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../libs')))
import connexion
import json

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.python.keras.models import load_model

from planespotter import planespotter

from flask import Flask, request, url_for, send_from_directory, render_template, make_response, abort
from werkzeug import secure_filename

MODEL_LOCATION = '../models'
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

import config

def create(filename):
    # print(config.ps)
    """
    This function creates a new prediction
    based on image (path) provided
    :param filename:  path to the local filename being predicted
    :return:        201 on success, 406 on prediction already exists
    """
    try:
        # print('######## CALLED THE POST PREDICTION API')
        # print("filename="+filename['filename'])

        image_name = filename['filename']

        # Init Prediction Object
        # ps = planespotter(model_location=MODEL_LOCATION)    
        # print('######## CREATED the PS OBJECT ')
        source_img = os.path.join(UPLOAD_FOLDER, image_name)    

        # print('Source image is at: '+ source_img)
# 
        # print(config.ps)
        
        # Predict image
        config.ps.predict_image(source_img)
        # print('######## CALLED PREDICT_IMAGE ')
        # Save Image
        config.ps.save_image(dir_path=OUTPUT_FOLDER)
        # Save Metadata
        json_pred = config.ps.save_metadata(dir_path=source_img)
        
        # print(json_pred)
        print(json.dumps(json_pred, indent=4))
        
        json_filename = os.path.join(OUTPUT_FOLDER, 
                                     os.path.splitext(image_name)[0]+'.json')
        with open(json_filename, 'w') as outfile:
            json.dump(json_pred, outfile)
        
        # print('######## DONE ')
        
        return (json_pred)
        # return make_response(
        #     "{image_name} successfully created".format(image_name=image_name), 201
        # )
    except:
        abort(
            406,
            "Error creating {image_name}".format(image_name=image_name),
        )


def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

def show_image_html(original_img,output_img,json_pred):
       return '''
    <!doctype html>
    <title>Prediction Output</title>
    <h1>'''+output_img+'''</h1>
    <img src="'''+original_img+'''" alt="'''+original_img+'''" height="600" width="600">
    <img src="'''+output_img+'''" alt="'''+output_img+'''" height="600" width="600">
    '''+json_pred+'''
    ''' 

# Create the application instance
app = connexion.App(__name__, specification_dir='./')

# Read the swagger.yml file to configure the endpoints
app.add_api('swagger.yml')

# Create a URL route in our application for "/"
# @app.route('/', methods=['POST'])
# def post():
#     print('***************************')
#     file = request.files['file']
#     print(file)
#     print('***************************')

#     file = request.files['file']
#     if file and allowed_file(file.filename):
#         print('**found file', file.filename)
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         # for browser, add 'redirect' function on top of 'url_for'
        
#         # Predict image
#         image_dest = os.path.join(UPLOAD_FOLDER, filename)
#         ps = planespotter(model_location=MODEL_LOCATION)
#         ps.predict_image(image_dest)
#         # Save Image
#         ps.save_image(dir_path=app.config['OUTPUT_FOLDER'])
        
#         image_scored = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
#         json_pred = ps.save_metadata(dir_path=image_dest)
        
# #            json_pred = json_pred.replace('\\"',"\"")
        
#         print(json_pred)
#         print(json.dumps(json_pred, indent=4))
        
        
#         json_filename = os.path.join(app.config['OUTPUT_FOLDER'], 
#                                         os.path.splitext(filename)[0]+'.json')
#         with open(json_filename, 'w') as outfile:
#             json.dump(json_pred, outfile)
        
#         return show_image_html(image_dest,image_scored,json.dumps(json_pred))    

#     # return render_template('about.html')

@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """
    return render_template('home.html')

# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)