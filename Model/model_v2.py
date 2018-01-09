'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
(https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

Data directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import Model
import time

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'fc_model.h5'
train_data_dir = 'face_data/train'
validation_data_dir = 'face_data/validation'
nb_train_samples = 400
nb_validation_samples = 200
top_model_epochs = 100
fine_tune_epochs = 100
batch_size = 10


def save_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('features_train.npy', 'wb'),
            features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('features_validation.npy', 'wb'),
            features_validation)


def train_top_model():
    train_data = np.load(open('features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(open('features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=top_model_epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

def fine_tune_model():
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width,img_height,3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(input=base_model.input, output=top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)


print("Started program.")
curr = time.time()
save_features()
print("Saved features in " + str(time.time() - curr) + " seconds.")
curr = time.time()
train_top_model()
print("Trained top model in " + str(time.time() - curr) + " seconds.")
fine_tune_model()
print("Fine tuned model in " + str(time.time() - curr) + " seconds.")