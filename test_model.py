# name: predict_emotion_server
# made: multimedia communication lab.
# date: 18. 7. 3

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
import time
import numpy as np
import os, sys, logging, json, requests, cv2, threading


logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

model_path = 'model_8.hdf5'
model_mean = [[[118.80855]]]
classes = {0:'angry', 1:'fear', 2:'happy', 3:'neutral', 4:'sad', 5:'surprise'}
emotions = [e.upper() for e in list(classes.values())]

model = load_model(model_path)

valid_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True, 
)


def prepare(test_img):
    '''
    Usage: The first time to run server, loading utils into memory
    return: None
    '''
    valid_x, valid_y = [], []

    z = load_img(path = test_img, grayscale=True, target_size=(227,227), interpolation='nearest')
    q = np.asarray(z).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)

    valid_datagen.fit(valid_x_moveaxis)
    valid_datagen.mean = model_mean

    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False))
    logging.debug(prob[0], classes[np.argmax(prob[0])])
    
    logging.info('Server ready')
    

class ThreadCam (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        
    def run(self):
        live_cam(self.name)

def live_cam(threadName):
    while (webcam.isOpened()):
        if cv2.waitKey(1)&0xFF == ord('q'):  
            break

        ret,frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x,y,w,h) in faces:
            #faces = faces[y+5:y+h-5, x+5:x+w-5]
            face_image_gray = gray[y:y+h, x:x+w]
            out = cv2.resize(face_image_gray, (227, 227), interpolation = cv2.INTER_AREA)
            reply = classfy(out)
            print(reply)
            k, v = reply.keys(), reply.values()
            k, v = list(reply.keys()), list(reply.values())
            answer = k[v.index(max(v))]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3, 4, 0)
            cv2.putText(frame, answer, (x+w//4, y), font, 0.9, (255,255,0),2)
            #print(out)
        cv2.imshow('demo', frame)
        
def classfy(face):
    """
    Usage: Classfify into emotions with the prediction probability
    return: list of probabilities
    """
    valid_x, valid_y = [], []
    reply = {}
    
    tic = time.time()
    q = np.asarray(face).astype('float32')
    q = np.asarray([q])
    
    valid_x.append(q)
    valid_x = np.asarray(valid_x)
    valid_x_moveaxis = np.moveaxis(valid_x, 1, 3)  
    
    valid_y.append([0, 0, 0, 0, 0, 0])
    valid_y = np.asarray(valid_y)

    valid_datagen.fit(valid_x_moveaxis)
    valid_datagen.mean = model_mean

    prob = model.predict_generator(valid_datagen.flow(valid_x_moveaxis, valid_y, batch_size=1, shuffle=False))
    
    for i in range(0, len(emotions)):
        reply[emotions[i]] = prob[0][i] #reply formet: {"EMOTION": probability, ...}
    
    toc = time.time()
    logging.info('time to classfication elasped', toc-tic) 
    
    return reply
        

#_______main_______:
webcam = cv2.VideoCapture(0)
  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

out = 0

print('load: ', model_path)
prepare('bbb.png')
#webcam thread run
    
threadLock = threading.Lock()
threads = []

# Create new threads
thread1 = ThreadCam(1)
    
# Start new Threads
thread1.start()

# Add threads to thread list
threads.append(thread1)

# Wait for all threads to complete
for t in threads:
    t.join()
print ("Exiting Main Thread")





