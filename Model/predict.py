import numpy as np
from keras.models import load_model
from keras import optimizers
import cv2
import time

# dimensions of our images.
img_width, img_height = 150, 150

model_path = 'model.h5'
# input_path = 'face_data/train/clinton/0.jpg'      # Should output 0
input_path1 = 'face_data/train/Barrack Obama/0.jpg'          # Should output 1
input_path2 = 'face_data/train/Aishwarya Rai/5.jpg'          # Should output 1
input_path3 = 'face_data/train/Hillary Clinton/0.jpg'          # Should output 1
input_path4 = 'face_data/train/Michael Jackson/3.jpg'          # Should output 1

def predict(input_path):
    model = load_model(model_path)

    img = cv2.imread(input_path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = model.predict(img)

    print(prediction)

print("Started program.")
curr = time.time()
predict(input_path1)
predict(input_path2)
predict(input_path3)
predict(input_path4)
print("Predicted in " + str(time.time() - curr) + " seconds.")
