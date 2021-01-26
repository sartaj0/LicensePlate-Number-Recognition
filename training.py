from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import os

img_height = 100
img_width = 100
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2) # set validation split

train_path = os.path.sep.join(["E:", "dataset", "LicensePlateDetector", "CNN_letters_Dataset"])

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_path, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') 


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3), activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(35, activation='softmax'))

print(model.summary())


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
callbacks = [
    ModelCheckpoint("model.h5", save_best_only=True)
]

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=800 // batch_size, callbacks=callbacks)