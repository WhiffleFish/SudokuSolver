import numpy as np
import mnist
import tensorflow as tf
from tensorflow import keras

# Load Mnist Data
print('Loading Data...')
X_train, y_train, X_test, y_test = (mnist.train_images(),
                                    mnist.train_labels(),
                                    mnist.test_images(),
                                    mnist.test_labels())

# Normalize Data
X_train = (X_train/255)
X_test = (X_test/255)


# Augment images to account for cell boundaries
def augment_img(image):
    img = image.copy()

    top, right, bot, left = np.random.random(4) > 0.6
    if top:
        size = np.random.randint(0, 3)
        img[:size] = min(np.abs(np.random.normal(loc=0.5, scale=0.5)), 1)
    if right:
        size = np.random.randint(0, 3)
        img[:, 28-size:] = min(np.abs(np.random.normal(loc=0.5, scale=0.5)), 1)
    if bot:
        size = np.random.randint(0, 3)
        img[28-size:] = min(np.abs(np.random.normal(loc=0.5, scale=0.5)), 1)
    if left:
        size = np.random.randint(0, 3)
        img[:, :size] = min(np.abs(np.random.normal(loc=0.5, scale=0.5)), 1)

    return img


print('Augmenting Image Data...')
aug_X_train = [augment_img(X_train[i]) for i in range(X_train.shape[0])]
aug_X_train = np.array(aug_X_train)

X_train_input = aug_X_train.reshape(*aug_X_train.shape, 1)
X_test_input = X_test.reshape(*X_test.shape, 1)

# One-hot encode labels
y_train_onehot = tf.one_hot(y_train, depth=10)
y_test_onehot = tf.one_hot(y_test, depth=10)


# CONSTRUCT CNN MODEL
# Architecture from:
# https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                              activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                              activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation="softmax"))


optimizer = keras.optimizers.Adam(learning_rate=3.0E-4)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

epochs = 2
batch_size = 86

# TRAIN MODEL
history = model.fit(X_train_input, y_train_onehot,
                    batch_size=batch_size,
                    epochs=epochs, verbose=2,
                    validation_data=(X_test_input, y_test_onehot))

model.save(r'models\AUG_DigitRecognizerModel')
