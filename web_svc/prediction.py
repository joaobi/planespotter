import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../libs')))
from planespotter import planespotter

import json
from flask import make_response, abort

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static'
MODEL_LOCATION = '../models'

def read():
    """
    This function responds to a request for /api/people
    with the complete lists of people

    :return:        sorted list of people
    """
    # Create the list of people from our data
    js = ''
    metadata = {}
    result = []
    for file in os.listdir(OUTPUT_FOLDER):
        if os.path.splitext(file)[1] == '.json':
            
            with open(os.path.join(OUTPUT_FOLDER,file), 'r') as jsonfile:
                js = json.load(jsonfile)
                # metadata[os.path.splitext(file)[0]] = js
                # metadata = {key: value for (key, value) in (metadata.items() + js.items())}
                metadata.update(js)
                result.append(js)
                # print(json.dumps(js, indent=2, sort_keys=True))
    
    # print(result)

    return (result)


def read_one(prediction_name):
    js = ''
    try:
        with open(os.path.join(OUTPUT_FOLDER,prediction_name+'.json'), 'r') as jsonfile:
            print('JSON FILE=>>'+prediction_name)
            js = json.load(jsonfile)
    except OSError as e:
        abort(
            404, "Prediction {predname} not found".format(predname=prediction_name)
        )
        print(e)
    return (js)

def create(filename):
    """
    This function creates a new prediction
    based on image (path) provided
    :param filename:  path to the local filename being predicted
    :return:        201 on success, 406 on prediction already exists
    """
    try:
        print('######## CALLED THE POST PREDICTION API')
        print("filename="+filename['filename'])

        image_name = filename['filename']

        # Init Prediction Object
        # ps = planespotter(model_location=MODEL_LOCATION)    
        print('######## CREATED the PS OBJECT ')
        source_img = os.path.join(UPLOAD_FOLDER, image_name)    

        print('Source image is at: '+ source_img)
        
        # Predict image
        ps.predict_image(source_img)
        # Save Image
        ps.save_image(dir_path=OUTPUT_FOLDER)
        # Save Metadata
        json_pred = ps.save_metadata(dir_path=source_img)
        
        print(json_pred)
        print(json.dumps(json_pred, indent=4))
        
        json_filename = os.path.join(OUTPUT_FOLDER, 
                                     os.path.splitext(image_name)[0]+'.json')
        with open(json_filename, 'w') as outfile:
            json.dump(json_pred, outfile)
        
        print('######## DONE ')
        
        return make_response(
            "{image_name} successfully created".format(image_name=image_name), 201
        )
    except:
        abort(
            406,
            "Error creating {image_name}".format(image_name=image_name),
        )