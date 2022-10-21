from cgi import test
import re
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import io
from model import *
import os
import glob
import cv2
import joblib
import matplotlib.pyplot as plt

def svmpredict(img_path, model=joblib.load('images/svm/t.pkl')):
    img = cv2.imread(img_path)

    x_train = np.load('images/svm/x_train.npy')
    print(feature_extractor(img=img, min_area=100))
    ncl_detect, error, ftrs = feature_extractor(img=img, min_area=100)
    if ncl_detect:
        ftrs = np.array(ftrs).reshape(1, -1)
        # normalize feature using max-min way
        mn, mx = x_train.min(axis=0), x_train.max(axis=0)
        ftrs = (ftrs - mn)/(mx - mn)
        print(ftrs)
        pred = model.predict(ftrs)
        return pred[0]
    else:
        return error

def load_image(image_file, svm=False):
    img = Image.open(image_file)
    # resize image to 2592x1944
    if not svm:
        
        image = img.resize((2592, 1944))
        # 24 bit color depth
        image = image.convert('RGB')
        image.save("images/tmp.jpg")
    else: 
        open_cv_image = np.array(img) 
        # open_cv_image = open_cv_image[:, :, ::-1].copy() 
        cv2.imwrite('images/svm/tmp.jpg', open_cv_image)
    # save image
    
def merge_image():
    images = [Image.open(x) for x in ['images/svm/nuc.jpg', 'images/svm/ROC.jpg']]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    st.image(new_im, caption='Links Nukleus Rechts Konvexe Hülle', use_column_width=True)

def remove_artifacts(img):
    image = cv2.imread(img)
    # crop 100 px from left and top
    image = image[100:, 100:]
    # crop 50 px bottom
    image = image[:-50, :]
    # crop 110 px right
    image = image[:, :-110]
    # overwrite image
    cv2.imwrite(img, image)

def read(prediction, test=False):
    if prediction == 1:
        prediction = 'Neutrophil'
    elif prediction == 2:
        prediction = 'Lymphozyt'
    elif prediction == 3:
        prediction = 'Monozyt'
    elif prediction == 4:
        prediction = 'Eosinophil'
    elif prediction == 5:
        prediction = 'Basophil'
    else:
        prediction = 'Keine Blutzelle gefunden'
    if test:
        return prediction
    else:
        st.write(f'Ergebnis: {prediction}')

def large_img_det(img, image_name='tmp'):
    image = cv2.imread(img)
    # Iterate over the image in 600x600 pixel steps, crop out each 600x600 pixel image and predict it
    # save the predictions in a list
    predictions = []
    for i in range(0, image.shape[0], 600):
        for j in range(0, image.shape[1], 600):
            # crop image
            crop_img = image[i:i+600, j:j+600]
            # save image
            cv2.imwrite(f'images/svm/{image_name}.jpg', crop_img)
            # predict
            predictions.append(read(svmpredict(img_path=f'images/svm/{image_name}.jpg'), test=True))
    # return the most common prediction
    common = max(set(predictions), key=predictions.count)
    st.write(f'Ergebnis: {common}')
    st.write(predictions)
