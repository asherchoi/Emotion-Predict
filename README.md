# Emotion-Predict
## A Convolutional Neural Network Model for Real-time Facial Expression Recognition
A convolutional neural network model for live facial expression recognition using traditional webcams is proposed so as to support up to 30 fps video sequence. Since we encounter a variety of noisy perceptible environments due the fact of distortion of facial images according to shooting angle and challenges associated with varying illumination conditions, suitable image pre-processing techniques are experimented and applied. To guarantee real-time prediction of video sequence, considering the trade-off between accuracy and processing time, different model structures and their hyper-parameters are employed and tested. Our experiments on the selected model show 95% accuracy for distorted images and the average recognition delay at 0.03sec per image. 

## Author
Chung Jae won, Choi Tae Hun, and Jin Pyo Hong, Department of Information and Communications Engineering Hankuk University of Foreign Studies

# System Architecture
![](https://github.com/asherchoi/Emotion-Predict/blob/master/system%20architecture.PNG)

## System Demonstration
![](https://github.com/asherchoi/Emotion-Predict/blob/master/demo.png)

## Development Environment
#### Client part
  + PL : C#
  + Source : [client](https://github.com/asherchoi/emotion-predict-client)
  + OS : Window10 64bit
  + Hardware : Microsoft kinect v2 moiton senser wired
  + Reference SDK : [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561)

#### Server part
  - PL : Anaconda with Python3.7.2
  - Source : [kinect_server.py](https://github.com/asherchoi/emotion-predict-server/blob/master/kinect_server.py)
  - OS : Linux Ubuntu 16.04 LTS 64bit
  - Deep Learning Framework: Keras 3.0 Tensorflow backend
  - Deep Learning Library & Package : Scikit learn 0.21.1 / Numpy 1.16.2 / Matplotlib 3.0.3 / Pandas 0.24.2

