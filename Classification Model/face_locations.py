import face_recognition
from PIL import Image

def crop(filename):
    image = face_recognition.load_image_file(filename)

    # Assumes here is only one face in the image
    face_locations = face_recognition.face_locations(image, model="cnn")
    print(face_locations)
    if face_locations:
        face_location = list(face_locations[0])

        print(face_location)
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        img = Image.fromarray(face_image)
        img.save("test.jpg")
        return face_location

#print(crop('face_data/train/Aishwarya Rai/73.jpg'))
print(crop('face_data/train/Barrack Obama/22.jpg'))
