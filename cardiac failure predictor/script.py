import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv("heart_failure.csv")
#print(data.info())
#print(Counter(data["death_event"]))
y = data['death_event']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

x = pd.get_dummies(x)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
 
numeric_features = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']

ct = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features)], remainder='passthrough')

X_train_scaled = ct.fit_transform(X_train)
X_test_scaled = ct.transform(X_test)

le = LabelEncoder()

Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))

Y_train = tensorflow.keras.utils.to_categorical(Y_train, dtype= 'int64')
#print(Y_train)
Y_test = tensorflow.keras.utils.to_categorical(Y_test, dtype = 'int64')
print(np.unique(Y_train).size)

model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_scaled, Y_train, epochs = 100, batch_size = 16, verbose = 0)

loss, acc = model.evaluate(X_train_scaled, Y_train, verbose=0)
print("Loss", loss, "Accuracy:", acc)

y_estimate = model.predict(X_test_scaled)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)
print(classification_report(y_true, y_estimate))
