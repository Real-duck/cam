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


from helper import *
# ---
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def cbc():
    st.sidebar.markdown('___')
    st.sidebar.markdown('# Blutkörperchen:')
    cell_type = st.sidebar.radio('', ['Rote Blutkörperchen', 'Weiße Blutkörperchen', 'Plätchen'])

    #
    if cell_type == 'Rote Blutkörperchen':
        cell_type = 'rbc'
    elif cell_type == 'Weiße Blutkörperchen':
        cell_type = 'wbc'
    else:
        cell_type = 'plt'

    count_it = st.sidebar.checkbox('Zählen?', value=False)
    st.sidebar.markdown('___')

    st.sidebar.markdown('# Einstellungen:')
    if cell_type == 'rbc':
        cht = st.sidebar.checkbox('Circle Hough Transform', value=True)
    else:
        cht = st.sidebar.checkbox('Circle Hough Transform', value=False)
        if cht: st.sidebar.warning('Circle Hough Transform gibt bei Plätchen und weißen Blutzellen schlechte Ergebnisse.')

    cl = st.sidebar.checkbox('Component Labeling', value=False)
    th = st.sidebar.checkbox('Threshold', value=False)

    selection_up = st.selectbox('Upload oder Testbild?', ['Testbild', 'Upload'])
    if selection_up == 'Upload':
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        st.warning('Bilder, die nicht mit einer PowerShot G5 camera gemacht wurden, können zu schlechten Ergebnissen führen.\n')

        if image_file is not None:
                load_image(image_file)
                file_details = {"filename":image_file.name, "filetype":image_file.type,
                                "filesize":image_file.size}
                st.write(file_details)

                image = "tmp"
                st.image(Image.open('images/' + image + '.jpg'), caption='Bild', use_column_width=True)
    else:
        st.markdown('___')
        st.markdown('# Bild auswählen:')
        image = st.radio('', ['Bild1', 'Bild2', 'Bild3', 'Bild4', 'Bild5'])
        st.image(Image.open('images/' + image + '.jpg'), caption='Testbild', use_column_width=True)
    if st.button('Start'):
        st_predict(img=f'images/{image}.jpg', cell_type=cell_type)
        st.warning('Das Zählen ist oft nicht korrekt.')
        if cell_type == 'rbc':
            remove_artifacts('images/edge_mask.png')
            if cht: hough_transform(img=f'images/edge_mask.png', cell_type='rbc')
            if cl: component_labeling(img=f'images/edge_mask.png')
            if th: stthreshold(img=f'images/edge_mask.png', cell_type='rbc')
            if count_it: 
                stthreshold(img=f'images/edge_mask.png', cell_type='rbc', show=False)
                stcount(img=f'images/threshold_edge_mask.jpg')
            
        else:
            if cht: hough_transform(img=f'images/mask.png', cell_type=cell_type)
            elif cl: component_labeling(img=f'images/mask.png')
            elif th: stthreshold(img=f'images/mask.png', cell_type=cell_type)
            elif count_it:
                stthreshold(img=f'images/mask.png', cell_type=cell_type, show=False)
                stcount(img=f'images/threshold_mask.jpg')
            stcount(img=f'images/mask.png', cell_type=cell_type)
