import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

training_data_generator = ImageDataGenerator(rescale=1/255)

train_iterator = training_data_generator.flow_from_directory('augmented-data/train',class_mode='categorical')

test_iterator = training_data_generator.flow_from_directory('augmented-data/test',class_mode='categorical', color_mode='grayscale',batch_size=16)

print("\nLoading validation data...")
print("\nBuilding model...")

my_model = tf.keras.Sequential()
my_model.add(tf.keras.Input(shape=(256, 256, 1)))
my_model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu"))
my_model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
my_model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu"))
my_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
my_model.add(tf.keras.layers.Flatten())
my_model.add(tf.keras.layers.Dense(3, activation="softmax"))
my_model.summary()

print("\nCompiling model")
my_model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.CategoricalCrossentropy(),
  metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)

print("\nTraining model...")

history = my_model.fit(
    train_iterator,
    steps_per_epoch=len(train_iterator),
    epochs=5,
    validation_data=train_iterator,
    validation_steps=len(train_iterator)
)

test_loss, test_accuracy = my_model.evaluate(test_iterator, steps=len(test_iterator), verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Do Matplotlib extension below

# use this savefig call at the end of your graph instead of using plt.show()
# plt.savefig('static/images/my_plots.png')
# Create subplots for Loss and AUC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting cross-entropy loss for both train and validation
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Cross-Entropy Loss Over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Plotting AUC for both train and validation
ax2.plot(history.history['auc'], label='Train AUC')
ax2.plot(history.history['val_auc'], label='Validation AUC')
ax2.set_title('AUC Over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('AUC')
ax2.legend()

# Save the figure
fig.savefig('static/images/my_plots.png')
