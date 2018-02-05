###########
# Imports #
###########

from flask import Flask, render_template, jsonify, redirect, request
from flask.ext.mysql import MySQL
from werkzeug.local import Local
import os
import numpy as np
from keras.models import load_model
import cv2
import face_recognition
from PIL import Image
import time

#########
# Setup #
#########

app = Flask(__name__)

if os.environ.get('TYPE') == 'production':
    app.config.from_object('config.ProductionConfig')
else:
    app.config.from_object('config.DevelopmentConfig')

mysql = MySQL()
mysql.init_app(app)

# dimensions of images
img_width, img_height = 150, 150

loc = Local()
loc.model = None

try:
    model_path = os.getcwd() + '/model.h5'

    classes_path = os.getcwd() + '/classes.txt'
    classes = open(classes_path).read().splitlines()
except:
    model_path = os.getcwd() + '\Server\model.h5'
    model = load_model(model_path)

    classes_path = os.getcwd() + '\Server\classes.txt'
    classes = open(classes_path).read().splitlines()

#############
# Endpoints #
#############

@app.route("/")
def home():
    if app.config.get('ENV') == "Dev":
        return "dev"
    elif app.config.get('ENV') == "Prod":
        return "prod"
    else:
        return "unknown"

@app.route("/only_predict", methods=['POST'])
def only_predict():
    if not loc.model:
        loc.model = load_model(model_path)

    img = cv2.imdecode(np.fromstring(request.files['image'].read(),
        np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = loc.model.predict(img)

    return classes[list(prediction[0]).index(1)]

@app.route("/predict", methods=['POST'])
def predict():
    img = cv2.imdecode(np.fromstring(request.files['image'].read(),
        np.uint8), cv2.IMREAD_COLOR)
    image = face_recognition.load_image_file(request.files['image'])

    face_locations = face_recognition.face_locations(image, model="cnn")

    filename = str(time.time()*1000) + ".jpg"
    cropped = False

    # Crop image to just the face if a face is detected
    if face_locations:
        cropped = True

        face_location = list(face_locations[0])

        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        image = Image.fromarray(face_image)
        image.save(filename)
        img = cv2.imread(filename)
        print("cropped")

    if not loc.model:
        loc.model = load_model(model_path)

    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = loc.model.predict(img)

    if cropped:
        os.remove(filename)

    return classes[list(prediction[0]).index(1)]


@app.route("/upload", methods=['GET'])
def upload():
    return render_template("upload.html")
