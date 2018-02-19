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

@app.route("/old_predict", methods=['POST'])
def old_predict():
    prediction = model.old_predict(request.files['image'])
    return prediction

@app.route("/predict", methods=['POST'])
def predict():
    prediction = model.predict(request.files['image'])
    return prediction

@app.route("/upload", methods=['GET'])
def upload():
    return render_template("upload.html")
