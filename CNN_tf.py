import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pyplot

from keras.preprocessing.image import ImageDataGenerator
# create generator
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory="/content/Earth_Images_Dataset/train_aug",
    target_size=(320, 240),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed = 42,
)

test_generator = test_datagen.flow_from_directory(
    directory="/content/Earth_Images_Dataset/test_aug",
    target_size=(320, 240),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed = 42,
)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, padding='same', activation='relu', input_shape=(320,240,3)))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = (1,1)))
model.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides = (1,1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=test_generator.n//test_generator.batch_size

history = model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=test_generator, 
                              validation_steps=STEP_SIZE_VALID, epochs=5)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# evaluate model
loss = model.evaluate_generator(test_generator, steps=1)
print('\n', 'Test accuracy:', loss[1])

# Take a look at the model summary
model.summary()
