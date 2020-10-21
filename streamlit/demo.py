import streamlit as st
from joblib import load
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.lite as tf
from urllib.request import urlopen

#loading data and model


@st.cache()
def load_data(file):
    return load(file)

X = load_data('./data/imgs.pkl')
y = load_data('./data/labels.pkl')

@st.cache()
def load_cnn():
    cnn = tf.Interpreter(model_path='./Saved Models/tuned_cnn.tflite')
    return cnn

cnn = load_cnn()

input_details = cnn.get_input_details()
output_details = cnn.get_output_details()
cnn.resize_tensor_input(input_details[0]['index'], (32, 128, 128, 1))
cnn.resize_tensor_input(output_details[0]['index'], (32, 1))
cnn.allocate_tensors()

#streamlit formatting
st.title('Car Break-in Detector Demo')
st.header('Description')
st.write('This demo shows the results of my car break-in detector with my final CNN model.')
st.write('A batch of 32 augmented images can be generated and predicted based on the parameters selected on the left sidebar.')
st.write('The Positive class is any image in the dataset that contains something resembling broken glass with or without a car.')
st.write('The Negative class is any image in the dataset that contains a car with no broken glass.')

st.header('Github:')
st.write('github.com/ian-andriot')

st.sidebar.header('Augmentation Parameters')

rotation_range = st.sidebar.slider('rotation', min_value=0, max_value=180)
width_shift = st.sidebar.slider('width shift', min_value=0., max_value=1.)
height_shift = st.sidebar.slider('height shift', min_value=0., max_value=1.)
zoom = st.sidebar.slider('zoom', min_value=0., max_value=1.)
horizontal_flip = st.sidebar.checkbox('Horizontal Flip')

image_gen = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift,
    height_shift_range=height_shift,
    zoom_range=zoom,
    horizontal_flip=horizontal_flip
    ).flow(x=X, y=y)

#generating and predicting images
if st.sidebar.button('Generate and Predict'):

    imgs, labels = image_gen.next()

    cnn.set_tensor(input_details[0]['index'], imgs)
    cnn.invoke()
    preds = np.round(cnn.get_tensor(output_details[0]['index'])).reshape(32)

    tp, tn, fp, fn = st.beta_columns(4)

    with tp:
        st.header('True Positives')
        st.write('Images correctly predicted to have broken glass')
        for i, img in enumerate(imgs):
            if labels[i] == preds[i] and labels[i] == 1.:
                st.image(img)

    with tn:
        st.header('True Negatives')
        st.write('Images correctly predicted to have no broken glass')
        for i, img in enumerate(imgs):
            if labels[i] == preds[i] and labels[i] == 0.:
                st.image(img)

    with fp:
        st.header('False Positives')
        st.write('Images incorrectly predicted to have broken glass')
        for i, img in enumerate(imgs):
            if labels[i] != preds[i] and labels[i] == 0.:
                st.image(img)
    
    with fn:
        st.header('False Negatives')
        st.write('Images incorrectly predicted to have no broken glass')
        for i, img in enumerate(imgs):
            if labels[i] != preds[i] and labels[i] == 1.:
                st.image(img)