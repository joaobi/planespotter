# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:07:59 2018

@author: joaobi
"""
import argparse
import os
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.python.keras import layers
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.keras.preprocessing import image

import numpy as np

# VARIABLES 
num_epochs = 75
learning_rate = 0.001 #0.001
model_name = 'models/airline_pred.h5'
model_json = 'models/airline_pred.json'

img_size = 200
#global labels
# labels = {}

parser = argparse.ArgumentParser(description='Train and Test a CNN to recognize\
                                 airlines')
parser.add_argument("-v", "--verbose",action="store_true",
                    dest="verbose", default=False,
                    help="print status messages to stdout")

parser.add_argument("-g", "--gui",action="store_true",
                    dest="gui", default=False,
                    help="print charts and images to GUI")


base_dir = ''
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

num_airlines = 6

def airline_model():
    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    img_input = layers.Input(shape=(img_size, img_size, 3))
    
    filter_size = 3 # was 3
    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(16, filter_size, activation='relu')(img_input)
    x = layers.MaxPooling2D(2)(x)
    
    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Conv2D(32, filter_size, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    x = layers.Convolution2D(64, filter_size, activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)

    # Flatten feature map to a 1-dim tensor
    x = layers.Flatten()(x)
    
    # Create a fully connected layer with ReLU activation and 512 hidden units
    x = layers.Dense(512, activation='relu')(x)
    
    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)
    
    # Create output layer with a x nodes (x=num clases) and softmax activation
    #output = layers.Dense(1, activation='sigmoid')(x)
    output = layers.Dense(num_airlines, activation='softmax')(x)
    
    # Configure and compile the model
    model = Model(img_input, output)
    
    return model

def augment():
    # Adding rescale, rotation_range, width_shift_range, height_shift_range,
    # shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)
    
    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            train_dir,  # This is the source directory for training images
            target_size=(img_size, img_size),  # All images will be resized to 150x150
            batch_size=20,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='sparse')
    
    # Flow validation images in batches of 32 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(img_size, img_size),
            batch_size=20,
            class_mode='sparse')
    
    global labels
    labels = (validation_generator.class_indices) 
    labels = dict((v,k) for k,v in labels.items())
    print(labels)

    return train_generator,validation_generator

def loss_accuracy_charts(history):
    # Retrieve a list of accuracy results on training and test data
    # sets for each training epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Get number of epochs
    epochs = range(len(acc))
    
    plt.figure()
    
    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    
    plt.figure()
    
    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

    plt.show()

def predict(model):
    image_name = 'val/SQ/SQ-#1825330.jpg'

    model = load_model(model_name)
    model.load_weights(model_name)

    img = image.load_img(image_name, target_size=(img_size, img_size))

    x = image.img_to_array(img)
    x = x / 255  
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)

    for k,pred in np.ndenumerate(preds[0]):
        print('Probability %s => [%0.2f%%]' % (labels[k[0]], pred*100))


if __name__== "__main__":
    args = parser.parse_args()

    # Build the model    
    model = airline_model()
    # Print model summary
    if args.verbose:
        model.summary()

    # from optimizers import optimizers

    from tensorflow.python.keras import optimizers

    lrate = 0.001
    decay = lrate/num_epochs
    sgd = optimizers.SGD(lr=lrate, momentum=0.90, decay=decay, nesterov=False)

    #Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'])
    
    train_generator,validation_generator = augment()
    
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=1300,  # 2000 images = batch_size * steps = 100
      epochs=num_epochs,
      validation_data=validation_generator,
      validation_steps=170,  # 1000 images = batch_size * steps = 50
      shuffle=True,
      verbose=1)
    

    # Save Model
    with open(model_json, "w") as json_file:
        json_file.write(model.to_json())
    model.save(model_name)
    
    # Print Accuracy charts
    if args.gui:
        loss_accuracy_charts()

    predict(model)
