import numpy as np
from keras.models import load_model
from keras import optimizers
import cv2
import time

# dimensions of our images.
img_width, img_height = 150, 150

model_path = 'model.h5'

classes_path = 'classes.txt'
classes = open(classes_path).read().splitlines()

# input_path1 = 'face_data/train/Barrack Obama/0.jpg'          # Should output 1
# input_path2 = 'face_data/train/Aishwarya Rai/5.jpg'          # Should output 1
# input_path3 = 'face_data/train/Hillary Clinton/0.jpg'          # Should output 1
# input_path4 = 'face_data/train/Michael Jackson/3.jpg'          # Should output 1
# input_path5 = 'face_data/train/Barrack Obama/1.jpg'          # Should output 1
# input_path6 = 'face_data/train/Aishwarya Rai/21.jpg'          # Should output 1
# input_path7 = 'face_data/train/Hillary Clinton/1.jpg'          # Should output 1
# input_path8 = 'face_data/train/Michael Jackson/12.jpg'          # Should output 1

# input_path1 = 'face_data/train/Barrack Obama/0.jpg'          # Should output 1
# input_path2 = 'face_data/train/Barrack Obama/1.jpg'          # Should output 1
# input_path3 = 'face_data/train/Barrack Obama/3.jpg'          # Should output 1
# input_path4 = 'face_data/train/Barrack Obama/10.jpg'          # Should output 1
# input_path5 = 'face_data/train/Barrack Obama/12.jpg'          # Should output 1
# input_path6 = 'face_data/train/Barrack Obama/14.jpg'          # Should output 1
# input_path7 = 'face_data/train/Barrack Obama/16.jpg'          # Should output 1
# input_path8 = 'face_data/train/Barrack Obama/18.jpg'          # Should output 1

# input_path1 = 'face_data/train/Michael Jackson/3.jpg'          # Should output 1
# input_path2 = 'face_data/train/Michael Jackson/12.jpg'          # Should output 1
# input_path3 = 'face_data/train/Michael Jackson/17.jpg'          # Should output 1
# input_path4 = 'face_data/train/Michael Jackson/29.jpg'          # Should output 1
# input_path5 = 'face_data/train/Michael Jackson/44.jpg'          # Should output 1
# input_path6 = 'face_data/train/Michael Jackson/45.jpg'          # Should output 1
# input_path7 = 'face_data/train/Michael Jackson/49.jpg'          # Should output 1
# input_path8 = 'face_data/train/Michael Jackson/57.jpg'          # Should output 1

# input_path1 = 'face_data/train/Hillary Clinton/0.jpg'          # Should output 1
# input_path2 = 'face_data/train/Hillary Clinton/1.jpg'          # Should output 1
# input_path3 = 'face_data/train/Hillary Clinton/3.jpg'          # Should output 1
# input_path4 = 'face_data/train/Hillary Clinton/5.jpg'          # Should output 1
# input_path5 = 'face_data/train/Hillary Clinton/6.jpg'          # Should output 1
# input_path6 = 'face_data/train/Hillary Clinton/8.jpg'          # Should output 1
# input_path7 = 'face_data/train/Hillary Clinton/10.jpg'          # Should output 1
# input_path8 = 'face_data/train/Hillary Clinton/11.jpg'          # Should output 1

input_path1 = 'face_data/train/Aishwarya Rai/5.jpg'          # Should output 1
input_path2 = 'face_data/train/Aishwarya Rai/21.jpg'          # Should output 1
input_path3 = 'face_data/train/Aishwarya Rai/31.jpg'          # Should output 1
input_path4 = 'face_data/train/Aishwarya Rai/34.jpg'          # Should output 1
input_path5 = 'face_data/train/Aishwarya Rai/47.jpg'          # Should output 1
input_path6 = 'face_data/train/Aishwarya Rai/58.jpg'          # Should output 1
input_path7 = 'face_data/train/Aishwarya Rai/73.jpg'          # Should output 1
input_path8 = 'face_data/train/Aishwarya Rai/74.jpg'          # Should output 1


def predict(input_path):
    model = load_model(model_path)

    img = cv2.imread(input_path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.reshape(img, [1, img_width, img_height, 3])

    prediction = model.predict(img)

    prediction = [int(round(i, 0)) for i in list(prediction[0])]
    print(prediction)
    print(classes[prediction.index(1)])

print("Started program.")
curr = time.time()
predict(input_path1)
predict(input_path2)
predict(input_path3)
predict(input_path4)
predict(input_path5)
predict(input_path6)
predict(input_path7)
predict(input_path8)
print("Predicted in " + str(time.time() - curr) + " seconds.")
