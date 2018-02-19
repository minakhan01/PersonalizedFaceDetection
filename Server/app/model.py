import os
import numpy as np
from keras.models import load_model
import cv2
try:
    import face_recognition
    using_face_recognition = True
except:
    using_face_recognition = False
import time

# dimensions of images
img_width, img_height = 150, 150

try:
    model_path = os.getcwd() + '/model.h5'

    classes_path = os.getcwd() + '/classes.txt'
    classes = open(classes_path).read().splitlines()
except:
    model_path = os.getcwd() + '\Server\model.h5'
    model = load_model(model_path)

    classes_path = os.getcwd() + '\Server\classes.txt'
    classes = open(classes_path).read().splitlines()

def predict(file):
    start_time = time.time()

    if using_face_recognition:
        image = face_recognition.load_image_file(file)
        image = cv2.resize(image, (img_width, img_height))
        locations = face_recognition.face_locations(image, model="cnn")

        # Crop image to just the face if a face is detected
        if locations:
            location = list(locations[0])
            top, right, bottom, left = location
            image = image[top:bottom, left:right]
    else:
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_width, img_height))

    #TODO: make it so that the model isn't loaded with every call
    model = load_model(model_path)

    image = cv2.resize(image, (img_width, img_height))
    image = np.reshape(image, [1, img_width, img_height, 3])

    prediction = model.predict(image)

    print("Time: " + str(time.time() - start_time))

    return classes[list(prediction[0]).index(1)]

def old_predict(file):
    start_time = time.time()

    #TODO: make it so that the model isn't loaded with every call
    model = load_model(model_path)

    img = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = model.predict(img)
    print("Time: " + str(time.time() - start_time))

    return classes[list(prediction[0]).index(1)]