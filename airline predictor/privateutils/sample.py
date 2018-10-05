# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:17:31 2018

@author: joaobi


Create Samples for Training and test out of the images available

Model test if plane is from airline X or not

"""

import os
from tqdm import tqdm
import random
import shutil
from PIL import Image

#airlines = ['EK','SQ','NH','MH','TP','TG','DL','QF','KE','CA','CX']
#output_dir = "train"
#airlines = ['EK','SQ']
#output_dir = "train_EKSQ/"
airlines = ['OZ','KE']
output_dir = "train_OZKE/"
num_samples_per_airline = 5000 ## was 5k


# Get 5k images per airline
train_test_ds = []
#for this_airline in airlines:
for this_airline in tqdm(airlines, total=len(airlines)):
    path = this_airline + "/"
    this_airline_images = os.listdir(path)
    this_airline_images  = [path+x for x in this_airline_images] #Add full path
    
    if len(this_airline_images) < num_samples_per_airline:
        num_samples = len(this_airline_images) 
    else:
        num_samples = num_samples_per_airline

    # Remove subdirs from total dataset
    for i in this_airline_images:
        if os.path.isdir(i):
#        print(i)
            this_airline_images.remove(i)

#    print("[%s] Picking %d samples out of %d"%(this_airline,
#          num_samples,len(this_airline_images)))
    
    # Remove portrait images
    for img in this_airline_images:
        im = Image.open(img)
        if im.size[0] < im.size[1]:
            this_airline_images.remove(img)
    
    this_airline_images = random.sample(this_airline_images,num_samples)
    train_test_ds.extend(this_airline_images)

# Copy file to train folder    
for file in tqdm(train_test_ds, total=len(train_test_ds)):
    shutil.copyfile(file, output_dir+file[:2]+'-'+file[3:])