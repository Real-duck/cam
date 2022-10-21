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
import base64
import pandas as pd


from model import *
from helper import *
# ---
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def create_csv():
    df = pd.DataFrame(columns=['Row', 'Cell', 'CHT', 'CL', 'Watershed', 'Count'])
    return df

def process_image(image, cell_type, c):
    values  = {'Row': c, 'Cell': cell_type, 'CHT': None, 'CL': None, 'Watershed': None, 'Count': None}


    image = 'tmp'
    st_predict(img=f'images/{image}.jpg', cell_type=cell_type)
    if cell_type == 'rbc':
        st.markdown('___')
        cht  = hough_transform(img=f'images/edge_mask.png', cell_type=cell_type)
        count = stcount(img=f'images/edge_mask.png', cell_type=cell_type)
        st.markdown('___')
        cl = None
        values['CHT'] = cht
        values['Count'] = count

    elif cell_type == 'wbc':
        st.markdown('___')
        count = stcount(img=f'images/mask.png', cell_type='wbc')
        cht = None
        cl = None
        st.markdown('___')
        values['Count'] = count
    else:
        st.markdown('___')
        cht = None
        count = None
        cl = component_labeling(img=f'imag{image}.jpg')
        values['Count'] = count
    return [c, cell_type, cht, cl, None, count]

st.title('Kamera')
df = create_csv()
df_val = []
img_file_buffer = st.camera_input("Take a picture")
cell_type=st.sidebar.selectbox('Blutzellen:', ['Rote Blutkörperchen', 'Weiße Blutkörperchen', 'Plätchen'], index=0)
if img_file_buffer is not None:

    load_image(img_file_buffer)
    if cell_type == 'Rote Blutkörperchen':
        cell_type = 'rbc'
    elif cell_type == 'Weiße Blutkörperchen':
        cell_type = 'wbc'
    else:
        cell_type = 'plt'
    
    # create dictionary for dataframe
    values = process_image(img_file_buffer, cell_type,  c=1)
    df_val.append(values)
st.write(df_val)
if st.button('Save'):
    df = pd.DataFrame(df_val, columns=['Row', 'Cell', 'CHT', 'CL', 'Watershed', 'Count'])
    df.to_csv('data.csv', index=False)
    st.write(df)  