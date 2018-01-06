###########
# Imports #
###########

from flask import Flask, render_template, jsonify, redirect
from flask.ext.mysql import MySQL
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