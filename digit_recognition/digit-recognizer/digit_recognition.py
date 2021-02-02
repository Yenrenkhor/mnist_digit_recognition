import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
image_index = 7777
print(y_train[image_index])
plt.imshow(X_train[image_index], cmap='Greys')
plt.show()

# Reshaping the array to 4-dims so that it can work with Keras API
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalizing the RGB code by dividing it to the max RGB value
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print('Number of image of X_train', X_train.shape[0])
print('Number of image of X_test', X_test.shape[0])

# Building the CNN

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# Flattening the 2D arrays for fully connected layers
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=tf.nn.softmax))

# Compiling and Fitting the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=128, epochs=10)

model.evaluate(X_test, y_test)

save_model(model, "../../digit_web/model/")

# model_json = model.to_json()
#
# with open("../../digit_web/model/model.json", "w") as json_file:
#     json_file.write(model_json)
#
# model.save_weights("../../digit_web/model/model.h5")

