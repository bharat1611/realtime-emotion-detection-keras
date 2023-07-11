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
# for simplicity, add the dataset in the same folder as your project and write:
# df = pd.read_csv("fer2013.csv")
print(df.info())