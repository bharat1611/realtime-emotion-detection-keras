import sys
import os
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import to_categorical

df = pd.read_csv("C:/Users/Acer/Desktop/DS Projects/Datasets/fer2013.csv")
# for simplicity, add the dataset in the same folder as your main.py and write:
# df = pd.read_csv("fer2013.csv")
# print(df.info())

X_train,train_y,X_test,test_y = [],[],[],[]

for index,row in df.iterrows():
    val = row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            test_y.append(row['emotion'])
    except:
        print(f"Error occurred at index : {index} and row {row}")

print(f"X_train sample data : {X_train[0:3]}")
print(f"train_y sample data : {train_y[0:3]}")
print(f"X_test sample data : {X_test[0:3]}")
print(f"test_y sample data : {test_y[0:3]}")

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

# Normalising the Data between 0 and 1
# We'll calculate the mean of the data and divide it by the std deviation
X_train = np.mean(X_train, axis = 0)
X_train /= np.std(X_train, axis = 0)

X_test = np.mean(X_test, axis = 0)
X_test /= np.std(X_test, axis = 0)