import os
import face_recognition
import cv2
from PIL import Image

input_directories = ['face_data/train', 'face_data/validation']
output_directories = ['cropped_face_data/train', 'cropped_face_data/validation']

img_width = 155, img_height = 155

for i, directory in enumerate(directories):
    for filename in os.listdir(directory):
        image = face_recognition.load_image_file(filename)

        # Assumes here is only one face in the image
        face_locations = face_recognition.face_locations(image, model="cnn")
        if face_locations:
            face_location = list(face_locations[0])

            top, right, bottom, left = face_location

            # You can access the actual face itself like this:
            image = image[top:bottom, left:right]
            print("cropped")
        image = Image.fromarray(image)
        image.resize(img_width, img_height)
        print(image.size)
        print(output_directories[i] + "/" + filename)
        image.save(output_directories[i] + "/" + filename)