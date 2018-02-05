import os
import face_recognition
import cv2
from PIL import Image

input_directories = ['face_data/train/Aishwarya Rai',
                     'face_data/train/Barrack Obama',
                     'face_data/train/Hillary Clinton',
                     'face_data/train/Michael Jackson',
                     'face_data/validation/Aishwarya Rai',
                     'face_data/validation/Barrack Obama',
                     'face_data/validation/Hillary Clinton',
                     'face_data/validation/Michael Jackson',]
output_directories = ['cropped_face_data/train/Aishwarya Rai',
                      'cropped_face_data/train/Barrack Obama',
                      'cropped_face_data/train/Hillary Clinton',
                      'cropped_face_data/train/Michael Jackson',
                      'cropped_face_data/validation/Aishwarya Rai',
                      'cropped_face_data/validation/Barrack Obama',
                      'cropped_face_data/validation/Hillary Clinton',
                      'cropped_face_data/validation/Michael Jackson',]

img_width = 155
img_height = 155

for i, directory in enumerate(input_directories):
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