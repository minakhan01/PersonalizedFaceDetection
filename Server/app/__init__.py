###########
# Imports #
###########

from flask import Flask, render_template, jsonify, redirect, request
from flask.ext.mysql import MySQL
import os
import numpy as np
from keras.models import load_model
import cv2


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

try:
    model_path = os.getcwd() + '/model.h5'
    model = load_model(model_path)

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
    img = cv2.imdecode(np.fromstring(request.files['image'].read(),
        np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = model.predict(img)

    return classes[list(prediction[0]).index(1)]

@app.route("/predict", methods=['POST'])
def predict(filename):
    image = face_recognition.load_image_file(request.files['image'])

    face_locations = face_recognition.face_locations(image, model="cnn")

    # Crop image to just the face if a face is detected
    if face_locations:
        face_location = list(face_locations[0])

        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]
        image = Image.fromarray(face_image)

    img = cv2.imdecode(np.fromstring(image.read(),
        np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = model.predict(img)

    return classes[list(prediction[0]).index(1)]


@app.route("/upload", methods=['GET'])
def upload():
    return render_template("upload.html")