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
from keras.utils import np_utils
import time, os, sys, operator

def save_features(img_width, img_height, batch_size, nb_train_samples,
        nb_validation_samples, train_data_dir, validation_data_dir):
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


def train_top_model(num_classes, nb_train_samples, nb_validation_samples,
        batch_size, top_model_epochs, top_model_weights_path):
    train_data = np.load(open('features_train.npy', 'rb'))
    train_labels = np_utils.to_categorical(np.array(
        [i for i in range(num_classes) for j in range(nb_train_samples // num_classes)]))
    # print(train_labels)

    validation_data = np.load(open('features_validation.npy', 'rb'))
    validation_labels = np_utils.to_categorical(np.array(
        [i for i in range(num_classes) for j in range(nb_validation_samples // num_classes)]))
    # print(validation_labels)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=top_model_epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

def fine_tune_model(img_width, img_height, num_classes, top_model_weights_path,
        train_data_dir, validation_data_dir, batch_size, fine_tune_epochs,
        model_path, classes_path, nb_train_samples, nb_validation_samples):
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False,
        input_shape=(img_width, img_height, 3))
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

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
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
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
        class_mode="categorical")

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical")

    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / batch_size,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples/batch_size)

    model.save(model_path)

    with open(classes_path, 'w') as file:
        for key, value in sorted(train_generator.class_indices.items(), key=operator.itemgetter(1)):
            file.write(key + "\n")

def train():
    # dimensions of our images.
    img_width, img_height = 150, 150

    top_model_weights_path = 'fc_model_weights.h5'
    model_path = 'model.h5'
    classes_path = 'classes.txt'
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    num_classes = sum(1 for i in os.listdir(train_data_dir))
    print(num_classes)
    nb_train_samples = 200 * num_classes
    nb_validation_samples = 100 * num_classes
    top_model_epochs = 100
    fine_tune_epochs = 100
    batch_size = 10

    print("Started program.")
    curr = time.time()
    save_features(img_width, img_height, batch_size, nb_train_samples,
        nb_validation_samples, train_data_dir, validation_data_dir)
    print("Saved features in " + str(time.time() - curr) + " seconds.")
    curr = time.time()
    train_top_model(num_classes, nb_train_samples, nb_validation_samples,
        batch_size, top_model_epochs, top_model_weights_path)
    print("Trained top model in " + str(time.time() - curr) + " seconds.")
    curr = time.time()
    fine_tune_model(img_width, img_height, num_classes, top_model_weights_path,
        train_data_dir, validation_data_dir, batch_size, fine_tune_epochs,
        model_path, classes_path, nb_train_samples, nb_validation_samples)
    print("Fine tuned model in " + str(time.time() - curr) + " seconds.")

if __name__ == '__main__':
    train()