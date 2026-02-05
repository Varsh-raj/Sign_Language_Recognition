import cv2
import numpy as np
import mediapipe as mp
import pyttsx3 as ts

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return lh

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

alphabet = np.array(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','undo','space'])
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(alphabet.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('Complete3.h5')

def find_avg(lst):
    return (sum(lst)/len(lst))

def text2speech(st):
    engine = ts.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[1].id)
    engine.setProperty('rate',120)
    engine.say(st)
    engine.runAndWait()

#New detection variables
sentence = []
st = ""
counter = 0
blank_count = 0
threshold = 0.3
previous_num = -1
cap = cv2.VideoCapture(0)
#access mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        #read feed
        ret, frame = cap.read()

        #make detection
        image, results = mediapipe_detection(frame, holistic)

        #draw landmarks
        draw_landmarks(image, results)

        #Prediction logic
        check = extract_keypoints(results)
        keypoints = extract_keypoints(results).reshape(1,-1)
        
        res = model.predict(np.expand_dims(keypoints, axis=0))[0]

        if find_avg(check) != 0:
            blank_count = 0
            #viz logic
            if res[np.argmax(res)] > threshold:
                
                if len(sentence) > 0:
                    if alphabet[np.argmax(res)] != sentence[-1]:
                        sentence.append(alphabet[np.argmax(res)])
                else:
                    sentence.append(alphabet[np.argmax(res)]) 
                counter += 1 

                if counter == 10:
                    if len(st) > 0:
                        if alphabet[np.argmax(res)] == 'undo':
                            st = st[:(len(st)-1)]
                        elif alphabet[np.argmax(res)] == 'space':
                            st += " "
                            cv2.putText(image, 'Printed', (100, 30), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255, 255, 255) , 2, cv2.LINE_AA)
                        else:
                            st += alphabet[np.argmax(res)].lower()
                    else: 
                        st += alphabet[np.argmax(res)]

            if len(sentence) > 1:
                sentence = sentence[-1:]
                counter = 0 

            #if counter > 10:
            #    cv2.putText(image, 'printed', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255, 255, 255) , 2, cv2.LINE_AA)

            cv2.rectangle(image, (0,0), (90, 40), (0, 0, 0), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255, 255, 255) , 2, cv2.LINE_AA)
        else:
            blank_count += 1
        
        if blank_count == 10:
            text2speech(st)
            st = ""
            blank_count = 0
        cv2.rectangle(image, (0,480), (650, 430), (0, 0, 0), -1)
        cv2.putText(image, st, (3, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


         
        #show to screen
        cv2.imshow('feed', image)

        #break on condition
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

