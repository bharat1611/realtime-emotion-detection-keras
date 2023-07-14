# Realtime Emotion Detection Using Keras

### This model has been trained for 40 epochs and runs at 71.69% accuracy.
## How to run model:
Install the dependencies below using pip or conda:
* pip install numpy
* pip install pandas
* pip install tensorflow
* pip install keras
* pip install opencv-python
* pip install future (used for Tkinter)

Download HAAR-Cascade file from :
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 

#### Download the Dataset here:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

#### First run main.py to create and train the model:
```
python main.py
```
#### Then run UI.py for implementing the face-emotion recognition interface:
```
python UI.py
```


## Overview:
Emotions are an integral part of human communication and behavior, playing a vital role in our interactions and decision-making processes. The ability to accurately detect and interpret emotions in real-time has significant implications across various domains, including human-computer interaction, healthcare, marketing, and entertainment. Real-time emotion detection systems leverage advanced technologies to analyze facial expressions, vocal cues, or physiological signals and classify them into specific emotional states.

Traditionally, emotion detection relied on subjective assessments or self-reporting, which could be prone to bias or inaccuracies. However, with advancements in artificial intelligence and machine learning, particularly in the field of computer vision and natural language processing, real-time emotion detection has become more accessible and reliable

#### There are different methodologies that can be followed to detect emotions, but in this project, we'll be using Keras (Tensorflow) along with OpenCV and haar-cascade.

## Keras Framework:
Keras provides a high-level API for building and training neural networks, making it an ideal choice for developing real-time emotion detection systems. 
In recent years, deep learning frameworks like Keras have gained popularity due to their ease of use and powerful capabilities. 

## Procedure:
To build a real-time emotion detection system using Keras, we typically follow a two-step process: data collection and model training. First, we gather a large dataset of labeled facial images, where each image is associated with a specific emotion. This dataset serves as the foundation for training the deep learning model. 

After training the model, we can deploy it to perform real-time emotion detection. By utilizing techniques like face detection and tracking, we can continuously analyze facial expressions from live video streams or recorded video footage. This enables applications such as emotion-aware user interfaces, interactive virtual characters, or even real-time emotion monitoring in healthcare settings.

## Dataset:
This project has made use of FER (Facial Emotion Detection) dataset formed and compiled in the year 2013. The original FER2013 dataset in Kaggle is available as a single csv file, here. I had this converted into a dataset of images in the PNG format for training/testing. 

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). 
train.csv contains two columns, "emotion" and "pixels". The "emotion" column contains a numeric code ranging from 0 to 6, inclusive, for the emotion that is present in the image. 


