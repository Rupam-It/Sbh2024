import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define paths
train_path = "C:\\Users\\Rupam\\All_CODING\\sbh2024"
test_path = "C:\\Users\\Rupam\\All_CODING\\sbh2024"

# Image data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Load and preprocess test data
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

# Build the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=38, activation='softmax')  # Assuming there are 38 classes in PlantVillage dataset
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(training_set, epochs=10, validation_data=test_set)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_set)
print(f"Test Accuracy: {test_acc}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
