# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 22:57:56 2018

@author: joaobi
"""
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
#    from tensorflow.python.keras.preprocessing import image

model_name = 'models/6airlines_10epochs_200.h5'
# image_name = 'val/SQ/SQ-#1825330.jpg'

#The labels are built in alphabetic order
# in case I want to make sure I can run the below
# labels = ['EK','SQ','NH','QF','KE','OZ']
# labels.sort()
# labels = {k: labels.index(k) for k in labels}
labels = {0: 'EK', 1: 'KE', 2: 'NH', 3: 'OZ', 4:'QF', 5:'SQ'}

parser = argparse.ArgumentParser(description='Predict airline [EK and SQ]')

parser.add_argument("-q", "--quiet",
                    action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")
#parser.add_argument("-f", "--file",
#                    action="store_false", dest="filename", default='',
#                    help="Predict this image file")

parser.add_argument("-i","--image", help="Path to image", default='')
#arguments = vars(parser.parse_args())

parser.add_argument("-d","--imagedir", help="Path to images directory", default='')
arguments = vars(parser.parse_args())

#parser.add_argument('imagename', nargs='?', help='path to image file',type=str)

def predict_airline(image_name,model):


#    image_name = 'photos/SIA3_A330_SQ.jpg'

    img=Image.open(image_name)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    # Rotate the image if needed
    if img._getexif() != None:    
        exif=dict(img._getexif().items())
#        print(exif[orientation] )
    
        if exif[orientation] == 3:
            img=img.rotate(180, expand=True)
 
    photo = img.resize((150,150), Image.ANTIALIAS)
    x = np.array(photo)
    x = x / 255  
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)   

    plt.imshow(photo)
    plt.axis('off')
    plt.show()
    
    return preds


if __name__== "__main__":
    args = parser.parse_args()
    
    if args.image =='':
        image_name = 'val/SQ/SQ-#1825330.jpg'
    else:
        image_name = args.image

    model = load_model(model_name)
    model.load_weights(model_name)

    preds = predict_airline(image_name,model)

    for k,pred in np.ndenumerate(preds[0]):
#        print(pred)
#        print(labels[k[0]])
        print('Probability %s => [%0.2f%%]' % (labels[k[0]], pred*100))

#    plt.imshow(img)
#    plt.axis('off')
#    plt.show()


#    plt.imshow(photo)
#    plt.axis('off')
#    plt.show()
