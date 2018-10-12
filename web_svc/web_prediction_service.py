# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:15:17 2018

@author: joaobi
"""

#!flask/bin/python
import sys
import os
#sys.path.append("c:/projects/airlinesdetection/libs")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../libs')))


import json

from planespotter import planespotter

from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename

MODEL_LOCATION = '../models'
UPLOAD_FOLDER = 'uploads'
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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print('**found file', file.filename)
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # for browser, add 'redirect' function on top of 'url_for'
            
            # Predict image
            image_dest = os.path.join(UPLOAD_FOLDER, filename)
            ps = planespotter(model_location=MODEL_LOCATION)
            ps.predict_image(image_dest)
            # Save Image
            ps.save_image(dir_path=app.config['OUTPUT_FOLDER'])
            
            image_scored = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            json_pred = ps.save_metadata(dir_path=image_dest)
            
#            json_pred = json_pred.replace('\\"',"\"")
            
            print(json_pred)
            print(json.dumps(json_pred, indent=4))
            
            
            json_filename = os.path.join(app.config['OUTPUT_FOLDER'], 
                                         os.path.splitext(filename)[0]+'.json')
            with open(json_filename, 'w') as outfile:
                json.dump(json_pred, outfile)
            
            return show_image_html(image_dest,image_scored,json.dumps(json_pred))
#            return url_for('uploaded_file',
#                                    filename=filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    print(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
    app.run(debug=True)

    
#@app.route('/images', methods=['POST'])
#def processImages(request):
#    method = request.method.decode('utf-8').upper()
#    content_type = request.getHeader('content-type')
#
#    img = FieldStorage(
#        fp = request.content,
#        headers = request.getAllHeaders(),
#        environ = {'REQUEST_METHOD': method, 'CONTENT_TYPE': content_type})
#    name = secure_filename(img[b'datafile'].filename)
#
#    with open(name, 'wb') as fileOutput:
#        # fileOutput.write(img['datafile'].value)
#        fileOutput.write(request.args[b'datafile'][0])
#
#
#@app.route('/postjson', methods=['POST'])
#def post():
#    print(request.is_json)
#    content = request.get_json()
#    #print(content)
#    print(content['id'])
#    print(content['name'])
#    return 'JSON posted'
#app.run(host='0.0.0.0', port=5000)