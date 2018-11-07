# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:20:43 2018

@author: joaobi
"""
import warnings
import os
import sys
#sys.path.append("C:/projects/models/research/")
#sys.path.append("C:/projects/models/research/object_detection")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../libs')))

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.python.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags


# import keras.backend as K

#import time  
import tensorflow as tf
from object_detection.utils import ops as utils_ops
#import ops as utils_ops
from PIL import ImageDraw
import PIL.ImageFont as ImageFont
import json

PATH_TO_OUTPUT_DIR = 'output'
DEBUG = False

MODEL_DIR = 'models'
#    
# Prediction Model
#
predict_model_name = '6airlines_75epochs_200_3.h5'
#PREDICT_MODEL = os.path.join(MODEL_DIR,predict_model_name)
labels = {0: 'EK', 1: 'KE', 2: 'NH', 3: 'OZ', 4:'QF', 5:'SQ'}
airline_names = {0: 'Emirates', 1: 'Korean Air', 2: 'ANA', 3: 'Asiana', 4:'Qantas', 5:'Singapore Airlines'}
SIZE = 500 # Resize for better inference
photo_size = 200 # Size for loading into CNN
#
# Detection Model
#
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PLANE_DETECTION_THRESHOLD = 0.60 # Detect plans above this prob. threshold

class planespotter:
    def __init__(self, model_location = MODEL_DIR):
        # print('INIT')
        self._init_detector(detect_model_loc = 
                            os.path.join(model_location,PATH_TO_FROZEN_GRAPH))
        # print('Loaded Detector')
        self._init_predictor(pred_model_location =
                             os.path.join(model_location,predict_model_name))
        # print('Loaded Predictor')


    def  _init_detector(self, detect_model_loc):

        self.model_name = detect_model_loc
        
#        print(self.model_name)
        
        self._build_obj_detection_graph()
        
        # 1. Loaded Image and then 2. Loaded Image with bboxes on it
        self.image_np = []
        # Boudnding Boxes for the planes
        self.bbox = []
        # Array with the cropped images of individual planes
        self.cropped_planes = []
        self.plane_idxs = []
        self.output_dict = []

        self.session = tf.Session(graph=self.detection_graph)     
 
        self.image_name = ''
    
    def predict_image(self,image_name):
        # 1. Detect planes on the image
        self.detect_planes(image_name)
        # print('Detected planes')
        # 2. Predict the airline of each plane
        self.predict_airline()
        # print('Predicted Airline')
        # 3. Draw the bounding boxes for each plane with airline and prob.
        self.draw_custom_bounding_boxes()
        # print('Drew BBs')

        # 4. Clear to avoid errors
        # tf.reset_default_graph() # for being sure
        # K.clear_session()
        # import gc
        # gc.collect()
        # print('CLOSING SESSION')
        # self.session.close()
        # tf.reset_default_graph()

       
    """
        Detect Planes and Crop them in the provided image
    """
    def detect_planes(self,image_name):
        
        #
        # 0. Pre Process Image
        #
        self.image_name = image_name
        image = self._preprocess_image(image_name)
        #
        #  1. Detect all Planes on this image
        #
#        step_time = time.time()
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        (im_width, im_height) = image.size
        if image.format == 'PNG':
            image = image.convert('RGB')
        image_np =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8) 
        # Actual detection.
        output_dict = self._run_inference_for_single_image(image_np, self.detection_graph)
#        print("[.............Detect objects] --- %.2f seconds ---" % (time.time() - step_time)) 

        #Only run the inference for Planes (Coco class 5)
        plane_idxs = np.where(output_dict['detection_classes']==5)
        self.plane_idxs = plane_idxs
        self.output_dict = output_dict
        
        #
        #  2. Crop Planes on this image
        #
        im_width, im_height = image.size
        cropped_planes = []
        self.bbox = []
        at_least_one_ex = -1
        
        num_planes_image = len(output_dict['detection_boxes'][plane_idxs])
        num_planes_thresh = len(np.where(output_dict['detection_scores'][plane_idxs]
                    >PLANE_DETECTION_THRESHOLD)[0])

        # I will still pick the highest if there are more than 0 but none is
        #   greater than the threshold
        if  num_planes_image > 0 and num_planes_thresh == 0:
            at_least_one_ex = np.argmax(output_dict['detection_scores'][plane_idxs])
        
        for plane in range(0,num_planes_image):
            plane_acc_score = output_dict['detection_scores'][plane_idxs][plane]
            if (plane_acc_score>PLANE_DETECTION_THRESHOLD or plane == at_least_one_ex):

                ymin = output_dict['detection_boxes'][plane_idxs][plane][0]
                xmin = output_dict['detection_boxes'][plane_idxs][plane][1]
                ymax = output_dict['detection_boxes'][plane_idxs][plane][2]
                xmax = output_dict['detection_boxes'][plane_idxs][plane][3]
                xmargin = im_width*0.02
                ymargin = im_height*0.02
                
                area = (xmin * im_width-xmargin, 
                      ymin * im_height-ymargin,
                      xmax * im_width+xmargin,
                      ymax * im_height+ymargin)
        
                self.bbox.append([ymin,xmin,ymax,xmax])
                
                cropped_planes.append(image.crop(area))

        self.cropped_planes = cropped_planes   
        self.image_np = image_np

    def draw_custom_bounding_boxes(self):
        thickness = 1
        color = 'blue'
        bbox = self.bbox
        
        airlines = self.predicted_airline
        probs = self.predicted_prob
        
        try:
            font = ImageFont.truetype('arial.ttf', 9)
        except IOError:
            font = ImageFont.load_default()        
        
        final_image = Image.fromarray(self.image_np)

        draw = ImageDraw.Draw(final_image)
        im_width, im_height = final_image.size

#        print(airlines)

        for i in range(0,len(airlines)):
            ymin,xmin,ymax,xmax = bbox[i][0],bbox[i][1],bbox[i][2],bbox[i][3]
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=thickness, fill=color)
    
            text_bottom = top

            display_str = '%s (%.2f%%)'%(airlines[i],probs[i])

            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
    
            draw.rectangle(
                    [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
                fill='blue')
            draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill='black',
                    font=font)
            text_bottom -= text_height - 2 * margin
   
        (im_width, im_height) = final_image.size
        if final_image.format == 'PNG':
            final_image = final_image.convert('RGB')
        self.image_np =  np.array(final_image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

        
    def _run_inference_for_single_image(self,image, graph):
#      step_time = time.time()
        # global detection_graph
        with self.detection_graph.as_default():
          sess = self.session
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
#          print("[.............pre inference] --- %.2f seconds ---" % (time.time() - step_time)) 
          # Run inference
#          step_time = time.time()
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
#          print("[.............Inference] --- %.2f seconds ---" % (time.time() - step_time)) 
    
#          step_time = time.time()
          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
#          print("[.............Post Inference] --- %.2f seconds ---" % (time.time() - step_time)) 
        return output_dict  

    def _build_obj_detection_graph(self):
        # global detection_graph
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_name, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.detection_graph = detection_graph

    def _preprocess_image(self,image_name):
        img=Image.open(image_name)
    
        if img.format=='JPEG':
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            # Rotate the image if needed
            if img._getexif() != None:
                exif=dict(img._getexif().items())
                if orientation in exif.keys():
                    if exif[orientation] == 3:
                        img=img.rotate(180, expand=True)
        
        # resize to get better inference speeds
        basewidth = SIZE
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        
        return img
        
    #
    # Prediction Methods
    #
    def _init_predictor(self,pred_model_location):
        try:
            model = load_model(pred_model_location)
            model.load_weights(pred_model_location)
            model._make_predict_function()

            self.model = model
            self.preds = []
            self.predicted_airline = []
            self.predicted_prob = []
        except Exception as e:
            print("ERROR: _init_predictor "+ str(e))
        
    def predict_airline(self):
        self.preds = []
        self.predicted_airline  = []
        self.predicted_prob = []
        model = self.model
        img_array = self.cropped_planes
        
        try:

            for img in img_array:
                photo = img.resize((photo_size,photo_size), Image.ANTIALIAS)
                x = np.array(photo)
                x = x / 255.  
                x = np.expand_dims(x, axis=0)
                
                preds = model.predict(x)
            
                if DEBUG:
                    predicted_airline = labels[np.argmax(preds[0])]
                    prob = np.max(preds[0])
                    plt.title('Airline '+predicted_airline+' with acc. '+ str(prob))
                    plt.imshow(photo)
                    plt.axis('off')
                    plt.show()        

                predicted_airline = airline_names[np.argmax(preds[0])]
                prob = np.max(preds[0])*100
    
                self.preds.append(preds)
                self.predicted_airline.append(predicted_airline)
                self.predicted_prob.append(prob)
                # print(preds)

        except Exception as e:
            print("[ERROR] [predict_airline]: "+ str(e))

    def print_stats(self):
        print("#objects detected: ",self.output_dict['num_detections'])
        print("#planes detected: ",str(len(self.plane_idxs[0])))
        thresh_nparr = np.where(self.output_dict['detection_scores'][self.plane_idxs]>PLANE_DETECTION_THRESHOLD)[0]
        num_planes_thresh = len(thresh_nparr)
        print("#planes shown: ",str(num_planes_thresh))
        # print(self.preds)
        for i in range(len(self.preds)):
            # score = self.output_dict['detection_scores'][i]
            # print("----->[Plane BBox %s]: %.2f%%"%(str(i),score))
            # for j,score in np.ndenumerate(np.sort(self.preds[i][0])[::-1]):
            #     print('Probability %s => [%0.4f%%]' % (labels[j[0]]
            #             , score*100))                  
            plane  = sorted(dict(zip(labels,self.preds[i][0])).items(),key=lambda kv: kv[1])[::-1]
            for j,score in plane:
                print('Probability %s => [%0.4f%%]' % (labels[j]
                        , score))

    def print_image(self):
        IMAGE_SIZE = (12, 8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(self.image_np)
        plt.show()
    
    def save_image(self,dir_path=PATH_TO_OUTPUT_DIR):
        image = Image.fromarray(self.image_np)
        filename = os.path.split(self.image_name)[-1]
        image.save(os.path.join(dir_path, filename), 'JPEG',subsampling=0, quality=100)

    def save_metadata(self,dir_path=PATH_TO_OUTPUT_DIR):
        metadata = {}
        metadata['filename'] = os.path.split(self.image_name)[-1]
        metadata['num_detections'] = self.output_dict['num_detections']
        metadata['planes_detected'] = len(self.plane_idxs[0])
        thresh_nparr = np.where(self.output_dict['detection_scores'][self.plane_idxs]>PLANE_DETECTION_THRESHOLD)[0]
        num_planes_thresh = len(thresh_nparr)         
        metadata['planes_shown'] = num_planes_thresh
        metadata['detection_boxes'] = self.output_dict['detection_boxes'][self.plane_idxs].tolist()
        metadata['detection_scores'] = self.output_dict['detection_scores'][self.plane_idxs].tolist()

        metadata['planes'] = []
        for i in range(len(self.preds)):
            name = 'plane_'+str(i)
            tmp = {} 
            plane  = sorted(dict(zip(labels,self.preds[i][0])).items(),key=lambda kv: kv[1])[::-1]
            for j,score in plane:
                print('Probability %s => [%0.4f%%]' % (labels[j]
                        , score*100))
                # tmp['Airline_'+str(j)] = labels[j]
                # tmp['ProbAir_'+str(j)] = score*100
                tmp[labels[j]] = score*100

            # for j,score in np.ndenumerate(np.sort(self.preds[i][0])[::-1]):
            #     print('Probability %s => [%0.4f%%]' % (labels[j[0]]
            #             , score*100))
            #     tmp['Airline_'+str(j[0])] = labels[j[0]]
            #     tmp['ProbAir_'+str(j[0])] = score*100
            print(tmp)
            # metadata[name] = tmp
            print(json.dumps(tmp))
            # metadata['planes'].append({'Airline_0': 'EK', 'ProbAir_0': 0.8939682, 'Airline_4': 'QF', 'ProbAir_4': 0.09370398, 'Airline_3': 'OZ', 'ProbAir_3': 0.009453673, 'Airline_5': 'SQ', 'ProbAir_5': 0.0014314898, 'Airline_1': 'KE', 'ProbAir_1': 0.00076366, 'Airline_2': 'NH', 'ProbAir_2': 0.0006790766})
            metadata['planes'].append(json.dumps(tmp))
            
            # print(metadata)
            # print(json.dumps(metadata, indent=4))
        # print(metadata)

        metadata['predicted_airline'] = self.predicted_airline
        metadata['predicted_prob'] = self.predicted_prob

        print(metadata)
        # print(json.dumps(metadata, indent=4))

        return metadata
#        json_data = json.dumps(metadata)
#        return json_data
       