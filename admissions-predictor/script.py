import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')

dataset = dataset.drop(columns=['Serial No.'])

features = dataset.iloc[:, :7]

labels = dataset.iloc[:, -1]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state = 42)

numeric_features = features.select_dtypes(include=['float64','int64']).columns.tolist()

ct = ColumnTransformer(transformers=[('standardize', StandardScaler(), numeric_features)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

my_model = Sequential(name='admission_nn')
num_of_features = features.shape[1]
input = layers.InputLayer(input_shape=(num_of_features,))
my_model.add(input)
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(1))
Sequential.summary(my_model)
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)

my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = my_model.fit(features_train_scaled, labels_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)

predicted_values = my_model.predict(features_test_scaled)

r2 = r2_score(labels_test, predicted_values)
print(f'R-squared value: {r2}')

epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_mae = history.history['mae']
val_mae = history.history['val_mae']

# Create a figure object
fig = plt.figure(figsize=(14, 6))

# Plot Loss
plt.subplot(1, 2, 1) 
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2) 
plt.plot(epochs, train_mae, label='Training MAE')
plt.plot(epochs, val_mae, label='Validation MAE')
plt.title('Mean Absolute Error per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig('static/images/my_plots.png') 
