import os
from os import listdir
from fer import FER
import pandas as pd
import cv2


detector = FER()
detector = FER(mtcnn=True)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

ret, frame = capture.read()

if ret: 
    for i in range(0,14):
        cv2.imwrite(f'Testnum{i}.png', frame)
else: 
    print('Image not detected')

count = 0

folder_dir = (r"C:\Users\HP\OneDrive\Máy tính\AI")
for images in os.listdir(folder_dir):
    if (images.endswith(".png")):
        img = cv2.imread(images)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,  scaleFactor = 1.2, minNeighbors = 5)
        if len(faces) == 0:
            print('Face not found!')
        else:
            emotion, score = detector.top_emotion(img)
            print(emotion)
            if emotion in ['happy', 'surprise']:
                count += 1

print(count)

# for i in range(0,14):
#     cv2.imshow('webCam', f'Testnum{i}.png')


# for i in range(0,14):
#     gray = cv2.cvtColor(f'Testnum{i}.png', cv2.COLOR_BGR2GRAY)
#     if len(faces) != 0:
#         print('Face not found!')
#     else:
#         faces = face_cascade.detectMultiScale(gray,  scaleFactor = 1.2, minNeighbors = 5)
#         emotion, score = detector.top_emotion(f'Testnum{i}.png')
#         if emotion in ['happy', 'surprise']:
#             count += 1


# while True:
#     ret, frame = capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray,  scaleFactor = 1.2, minNeighbors = 5)
#     detector.detect_emotions(frame)
#     emotion, score = detector.top_emotion(frame)


#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     if len(faces) != 0: 
#         frame = cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     else: 
#         frame = cv2.putText(frame, 'Face not found', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.imshow('webCam',frame)
#     if (cv2.waitKey(1) == ord('s')):
        # break

# capture.release()
# cv2.destroyAllWindows()

