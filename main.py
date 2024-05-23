"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titanic_data = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

titanic_data.isnull().sum()
print("Train Shape:", titanic_data.shape)
titanic_test.isnull().sum()
print("Test Shape:", titanic_test.shape)

titanic_data.head(10)

titanic_data.describe()
titanic_test.describe()
print(titanic_data.isnull().sum())

titanic_test.isnull().sum()
titanic_test["Survived"] = ""
titanic_test.head()


def bar_chart(feature):
    survived = titanic_data[titanic_data['Survived'] == 1][feature].value_counts()
    dead = titanic_data[titanic_data['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


bar_chart('Sex')
print("Survived :\n", titanic_data[titanic_data['Survived'] == 1]['Sex'].value_counts())
print("Dead:\n", titanic_data[titanic_data['Survived'] == 0]['Sex'].value_counts())
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm

import tensorflow as tf

data = pd.read_csv("train.csv")

# print(data)

columns_target = ['Survived']  # выжившие

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']

X_train = data[columns_train]
Y_train = data[columns_target]

X_train['Pclass'].isnull().sum()
X_train['Sex'].isnull().sum()
X_train['Age'].isnull().sum()
X_train['Fare'].isnull().sum()

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
# X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())

d = {'male': 0, 'female': 1}
X_train['Sex'] = X_train['Sex'].apply(lambda x: d[x])
print(X_train['Sex'].head())

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.67, random_state=42)
#
# predmodel = svm.LinearSVC()  # метод опорных векторов
# predmodel.fit(X_train, Y_train)
# print(predmodel.predict(X_test[0:10]))
#
# print(predmodel.score(X_test, Y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='softmax')
])

# model.add(tf.keras.layers.Conv2D(32, (3,1), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(None, 1)))
# model.add(tf.keras.layers.LSTM(64, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='softmax'))

x_val = X_train[-10000:]
y_val = Y_train[-10000:]
# X_train = X_train[:-10000]
# Y_train = Y_train[:-10000]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=callback)

test = pd.read_csv("test.csv")

X_test = test[columns_train]
# Y_test = test[columns_target]

X_test['Pclass'].isnull().sum()
X_test['Sex'].isnull().sum()
X_test['Age'].isnull().sum()
X_test['Fare'].isnull().sum()

# X['Age'] = X['Age'].fillna(X['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

X_test['Sex'] = X_test['Sex'].apply(lambda x: d[x])
print(X_test['Sex'].head())

loss, accuracy = model.evaluate(X_test)

# model.save('my_model.h5')

print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

"""import pandas as pd
import numpy as np
import tensorflow as tf

test = pd.read_csv("test.csv")

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']
X_test = test[columns_train]
# Y_test = test[columns_target]

X_test['Pclass'].isnull().sum()
X_test['Sex'].isnull().sum()
X_test['Age'].isnull().sum()
X_test['Fare'].isnull().sum()

X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

d = {'male': 0, 'female': 1}
X_test['Sex'] = X_test['Sex'].apply(lambda x: d[x])
print(X_test['Sex'].head())

new_model = tf.keras.models.load_model('my_model.h5')
loss, accuracy = new_model.evaluate(X_test)

print('Test Loss:', loss)
print('Test Accuracy:', accuracy)"""