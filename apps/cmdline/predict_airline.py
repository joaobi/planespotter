# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:44:29 2018

@author: joaobi
"""
#import airline_utils
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../libs')))
from planespotter import planespotter

SHOW_IMAGE = False
SHOW_IMAGE_WINDOW_SPYDER = False
SHOW_STATISTICS_IMAGE = False
SHOW_STATISTICS_AGG = False
EXPORT_IMAGE = True

MODEL_LOCATION = '../../models'
INPUT_IMAGES_DIR = 'input_images'
INPUT_IMAGE_PATHS = os.listdir(INPUT_IMAGES_DIR)
OUTPUT_IMAGES_DIR = 'output_images'


INPUT_IMAGE_PATHS = ['1718180.jpg']

if __name__== "__main__":    

    start_time = time.time()
    
    ps = planespotter(model_location=MODEL_LOCATION)

    for image_path in INPUT_IMAGE_PATHS:
        image_time = time.time()

        ps.predict_image(os.path.join(INPUT_IMAGES_DIR, image_path))

#        # 1. Detect planes on the image
#        ps.detect_planes(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path))
#        # 2. Predict the airline of each plane
#        ps.predict_airline()
#        # 3. Draw the bounding boxes for each plane with airline and prob.
#        ps.draw_custom_bounding_boxes()

        # Show the images with the planes and their predicted Airline
        if SHOW_IMAGE:
            if SHOW_IMAGE_WINDOW_SPYDER:
                ps.final_image.show() # Show in Window outside Spyder
            else:
                ps.print_image()
        
        # Show Statisticsfor this image
        if SHOW_STATISTICS_IMAGE:
            ps.print_stats()
        
        if EXPORT_IMAGE:
            ps.save_image(dir_path=OUTPUT_IMAGES_DIR)
            print(ps.save_metadata(dir_path=OUTPUT_IMAGES_DIR))
        
        print("====>[%s] --- %.2f seconds ---" % (image_path,time.time() - image_time))
    
    # Show Aggregated staistics for this pass
    if SHOW_STATISTICS_AGG:
        pass

        
#    PATH_TO_TEST_IMAGES_DIR = 'test_images'
#    PATH_TO_TEST_IMAGES_DIR = 'downloads\\singapore airlines airplane'
#    TEST_IMAGE_PATHS = os.listdir(PATH_TO_TEST_IMAGES_DIR)
#    
#    # Load Plane detector
#    step_time = time.time()
#    pd = airline_utils.PlaneDetector()
#    detection_graph = pd.detection_graph
#    print("[BOOSTSTRAP PLANE DETECTOR ] --- %.2f seconds ---" % (time.time() - step_time))
#    
#    
##    TEST_IMAGE_PATHS = ['Asiana1_A330_OZ.jpg','Asiana2_B767_OZ.jpg']
##    TEST_IMAGE_PATHS = ['Asiana2_B767_OZ.jpg']
##    TEST_IMAGE_PATHS = ['4261861.jpg','1675022.jpg','1718180.jpg','2391507.jpg','2480083.jpg']
#    
##    TEST_IMAGE_PATHS = ['73. 33fa574500000578-3580633-image-a-2_1462786524096.jpg']
#    
#    # Load Plane Predictor
#    step_time = time.time()    
#    pp = airline_utils.PlanePredictor()
#    print("[BOOSTSTRAP PLANE PREDICTOR] --- %.2f seconds ---" % (time.time() - step_time))    
#    
#    for image_path in TEST_IMAGE_PATHS:
#        image_time = time.time()
#        step_time = time.time()
#        image = pp.preprocess_image(os.path.join(PATH_TO_TEST_IMAGES_DIR, image_path))
##        print("[PREPROCESS IMAGE] --- %.2f seconds ---" % (time.time() - step_time)) 
#
#        step_time = time.time()
#        pd.detect_planes(image)
##        print("[1. DETECT PLANES] --- %.2f seconds ---" % (time.time() - step_time)) 
#    
#        step_time = time.time()
#        pd.crop_plane()
##        print("[2. CROP IMAGES] --- %.2f seconds ---" % (time.time() - step_time)) 
#    
#        step_time = time.time()
#        pp.predict_airline(pd.cropped_planes)
##        print("[3. PREDICT AIRLINE] --- %.2f seconds ---" % (time.time() - step_time)) 
#
#        step_time = time.time()
#        pd.draw_custom_bounding_boxes(pp.predicted_airline,pp.predicted_prob)
##        print("[4. DRAW BOUNDING BOXES] --- %.2f seconds ---" % (time.time() - step_time)) 
#
#        # Show the images with the planes and their predicted Airline
#        if SHOW_IMAGE:
#            if SHOW_IMAGE_WINDOW_SPYDER:
#                pd.final_image.show() # Show in Window outside Spyder
#            else:
#                pd.print_image()
#        
#        # Show Statisticsfor this image
#        if SHOW_STATISTICS_IMAGE:
#            pd.print_detections()
#            pp.print_predictions()
#        print("====>[%s] --- %.2f seconds ---" % (image_path,time.time() - image_time))
#    
#    # Show Aggregated staistics for this pass
#    if SHOW_STATISTICS_AGG:
#        pass
    
    print("--- %.2f seconds ---" % (time.time() - start_time))
    

