# import numpy as np
import io
import os
import random
import shutil

import cv2
# from pipeline import *
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from classifiers import *
import face_recognition

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)


st.title('Deepfake Detector')

# images = glob.glob("")

dataGenerator = ImageDataGenerator(rescale=1. / 255)

import glob


#
# btnResult = st.form_submit_button('Donald')
# if btnResult:
#
#     path = './test_images/donald_r/*.*'


def predict(res1):
    generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=30,
        class_mode='binary',
        subset='training')

    # st.text('Ready for deeepfake prediction.......')
    num_to_label = {1: "REAL", 0: "FAKE"}

    # 3 - Predict
    X, y = generator.next()

    probabilistic_predictions = classifier.predict(X)

    print('Predicted :', probabilistic_predictions)
    predictions = []
    for x in probabilistic_predictions:
        if x[0] > 0.75:
            predictions.append(num_to_label[1])
        else:
            predictions.append(num_to_label[0])
    # predictions = [num_to_label[round(x[0])] for x in probabilistic_predictions]
    print(predictions)

    return predictions, probabilistic_predictions


def locating_face_landmarks(image):
    face_landmarks_list = face_recognition.face_landmarks(image)

    # https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py
    # face_landmarks_list
    from PIL import Image, ImageDraw
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature,
                                                                            face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=3)

    return pil_image


def detect_face(image, frame, new_path):

    face_locations = face_recognition.face_locations(image)

    # https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.grid(False)
        # ax.xaxis.set_visible(False)
        # ax.yaxis.set_visible(False)
        # ax.imshow(face_image)
        # st.image(face_image)

        cv2.imwrite(new_path + '/face_by_frame_' + str(frame) + '.jpg', face_image)
        # print('Now detecting face location.....')
        # locating_face_landmarks(face_image)


def extract_image(directory):
    directory = 'upload'

    for filename in os.listdir(directory):
        print(filename)
        new_path = 'extracted_images/' + str(os.listdir(directory).index(filename))
        print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    f = os.path.join(directory, filename)

    vid_cap = cv2.VideoCapture(f)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    thirty_rand = random.sample(list(range(0, total_frames)), 20)

    for frame in thirty_rand:
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        success, image = vid_cap.read()

        detect_face(image, frame, new_path)

    return new_path


option = st.selectbox(
    "How would you like to start",
    ("Choose dataset", "person_1", "upload video"))
source_dir = ''
path = ''

print(option)

if option != 'Choose dataset' and option != 'upload video':
    source_dir = f'datset/{option}'
    path = f'test_images/{option}'
    destination = f'test_images/{option}'
    img = cv2.imread(path)


def save_uploadedfile(file):
    g = io.BytesIO(file.read())  ## BytesIO Object
    temporary_location = f'upload/{file.name}'

    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file

    # close file
    out.close()
    print("Saved File:{} to upload".format(file.name))
    return temporary_location


if option == 'upload video':
    # video_file = st.file_uploader('video', type=['mp4'])
    # cap = cv2.VideoCapture(video_file)

    f = st.file_uploader("Upload video file")
    if f is not None:

        loc = save_uploadedfile(f)

        video_file = open(loc, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        source_dir = extract_image(loc)
        path = f'test_images/{option}'
        destination = f'test_images/{option}'


if path != '' and source_dir != '':
    shutil.copytree(source_dir, destination)
    res = glob.glob(path + '/*.*')
    # dir_list = os.listdir(path)
    print(res)
    # image1 = Image.open(path + image)
    predictions_labels, probabilistic_predictions = predict(res)

    st.text('Extracted face images from above video are .......')
    st.image(res, width=100)

    st.text('Detecting the face localisation and predicting for deep Fake.......')

    col1, col2, col3 = st.columns(3)

    for ind, im in enumerate(res):
        img = cv2.imread(im)
        with col1:
            if ind == 0:
                st.header("Extracted face ")
            st.image(img, width=200)

        with col2:
            if ind == 0:
                st.header("Localization")
            locating_face_image = locating_face_landmarks(img)
            st.image(locating_face_image, width=200)

        with col3:
            if ind == 0:
                st.header("FAKE/REAL")

            if predictions_labels[ind] == 'FAKE':
                pred_prob = str(round((1 - probabilistic_predictions[ind][0])*100, 2))
            else:
                pred_prob = str(round(probabilistic_predictions[ind][0] * 100, 2))
            text = predictions_labels[ind] + ' -->  ' + pred_prob + '% confidence'
            tabs_font_css = """
            <style>
            div[class*="stText"] label {
              font-size: 22px
            }
            </style>
            """
            st.write(tabs_font_css, unsafe_allow_html=True)

            st.text_area(text)

    shutil.rmtree(f'test_images/{option}')

    if option == 'upload video':
        shutil.rmtree(source_dir, 'completed')
        upload_file = 'upload/' + f.name
        shutil.move(upload_file, 'completed')
    source_dir = ''
    path = ''
    option = ''

# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F.h5')

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
