###########
# Imports #
###########

from flask import Flask, render_template, jsonify, redirect, request
from flask.ext.mysql import MySQL
from . import model
import os

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

# Takes in a single image, returns the prediction as plain text.
# Does not crop to a face with the face_recognition module
@app.route("/old_predict", methods=['POST'])
def old_predict():
    prediction = model.old_predict(request.files['image'])
    return prediction

# Takes in a single image, returns the prediction as plain text.
# Crops the image to a face with the face_recognition module
@app.route("/predict", methods=['POST'])
def predict():
    prediction = model.predict(request.files['image'])
    return prediction

# Allows the user to choose a file to predict.
# Only use this for testing on a computer. Call /predict or /old_predict
# directly from devices that send requests to the server.
@app.route("/upload", methods=['GET'])
def upload():
    return render_template("upload.html")

# Takes in 3 inputs a name, a set of 200 training images, and a set of 100
# validation images and updates the model with a new class based on the inputs.
# Needs to finish training before the request is finished, so each request takes
# at least 10 minutes.
@app.route("/train", methods=['POST'])
def train():
    train_files = request.files.getlist("train[]")
    validation_files = request.files.getlist("validation[]")
    model.add_class(request.form["name"], train_files, validation_files)
    return redirect('/upload_multiple')

# Allows the user to input a name, select 200 training iamges, and 100
# validation images.
# Only use this for testing on a computer. Call /train directly from devices
# that send requests to the server.
@app.route("/upload_multiple", methods=['GET'])
def upload_multiple():
    return render_template("upload_multiple.html")