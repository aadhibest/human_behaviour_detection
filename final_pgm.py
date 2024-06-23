# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:31:31 2020

@author: Aadhithya
"""

import cv2
import numpy as np
import person_and_phone as pnp
import eye_tracker as track
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

caffeModel = "C:/Users/Aadhithya/Desktop/Proctoring-AI-master/models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "C:/Users/Aadhithya/Desktop/Proctoring-AI-master/models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('Final result.avi', fourcc, 20.0, (640, 480))
face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
ret, image = cap.read()
thresh = image.copy()
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 75, 255, track.nothing)

kernel = np.ones((9, 9), np.uint8)
warning = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret, image = cap.read()
    ret1, frame = cap.read()
    if ret == False:
        break
    img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (320, 320))
    img1 = img1.astype(np.float32)
    img1 = np.expand_dims(img1, 0)
    img1 = img1 / 255
    class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
    boxes, scores, classes, nums = pnp.yolo(img1)
    count=0
    for i in range(nums[0]):
        if int(classes[0][i] == 0):
            count +=1
        if int(classes[0][i] == 67):
            print('Mobile Phone detected - Warning')
            warning += 1 
            if warning <=10:
                cv2.putText(frame, 'WARNING - Using Phone', (30, 30), font,  
                            1, (255, 0, 0), 2, cv2.LINE_AA)
    if count == 0:
        print('No person detected')
        warning += 1
        if warning <=10:
            cv2.putText(frame, 'WARNING - No face detected', (30, 30), font,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        
    elif count > 1: 
        print('More people detected - Warning')
        warning += 1
        if warning <=10:
            cv2.putText(frame, 'WARNING - More people detected', (30, 30), font,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    
    
    rects = find_faces(image, face_model)
    
    for rect in rects:
        shape = detect_marks(image, landmark_model, rect)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask, end_points_left = track.eye_on_mask(mask, left, shape)
        mask, end_points_right = track.eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(image, image, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = track.process_thresh(thresh)
        
        eye_left = track.contouring(thresh[:, 0:mid], mid, image, end_points_left)
        eye_right = track.contouring(thresh[:, mid:], mid, image, end_points_right, True)
        
        if eye_left == eye_right and eye_left != 0:
            text = ''
            if eye_left == 1:
                print('Looking Left - Warning')
                text = 'WARNING - Looking Left'
                warning += 1 
            elif eye_left == 2:
                print('Looking right - Warning')
                text = 'WARNING - Looking right'
                warning += 1
            #elif left == 3:
                #print('Looking up')
                #text = 'Looking up'
            if warning <=10:
                cv2.putText(frame, text, (30, 30), font,  
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    
    frame = cv2.resize(frame,(640,480))

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h, w) = frame.shape[:2]
    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()
    

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence and prediction

        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum confidence
        if confidence < 0.5 :
            continue

        # Determine the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        print(confidence)
        # draw the bounding box of the face along with the associated
        #text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (255, 0, 0), 2)
        #cv2.putText(frame, text, (startX, y),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
    #cv2.imshow("Frame", frame)
    
    if (warning >100) or (cv2.waitKey(1) & 0xFF==ord('c')):
        cv2.putText(frame, 'MALPRACTICE DETECTED', (100, 400), font,  
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Final', frame)
    cv2.imshow("Eye", image)
    out.write(frame)
    cv2.imshow("image", thresh)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (warning > 200):
        break


cap.release()
out.release()
cv2.destroyAllWindows()