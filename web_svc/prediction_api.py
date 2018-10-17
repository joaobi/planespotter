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

from planespotter import planespotter

from flask import Flask, request, url_for, send_from_directory, render_template
from werkzeug import secure_filename

MODEL_LOCATION = '../models'
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

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