import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import cifar10
from Conv_DCFD_tf2 import *

# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0



if True:
    model = models.Sequential([
        Conv_DCFD_tf(in_channels=3, out_channels=10, kernel_size=3),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        # layers.MaxPooling2D((2, 2)),
        # Conv_DCFD_tf(in_channels=10, out_channels=128, kernel_size=3),
        tf.keras.layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        layers.Flatten(),
        # layers.Dense(512, activation='relu'),
        
        layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
else:
    # Define the model
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=7,
                    validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

"""
Normal convolutions:
313/313 - 0s - loss: 0.8698 - accuracy: 0.7174 - 451ms/epoch - 1ms/step
Test accuracy: 0.7174000144004822

Test accuracy: 0.6970000267028809

"""