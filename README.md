# Human Behaviour Detection during Online Exams 

This project aims to detect and prevent malpractice during online exams by leveraging the webcam of the examinee's computer. The project uses deep learning algorithms for various aspects of detection, such as face detection, gaze detection, and object detection. 
<br>The models used for each of the detections are,
<br>For Face Detection: OpenCV's DNN model (SSD architecture)
<br>For Gaze Detection: OpenCV's DNN model (SSD architecture for detection and ResNet-10 CNN for feature extraction)
<br>For Object Detection: Yolov3 Model (Darknet-53 architecture)

## Working of the Final Program
<br>The Final program leverages all the three models to detect behaviour of the examinee during the exam in realtime.
<br>With Object detection, it detects the usage of any unapproved objects such as a mobile phone during the course of the exam.
<br>With Gaze detection, it detects if the examinee is looking away from the screen.
<br>With Face Detection, it detects if the candidate alone is present during the course of the exam, i.e, no one else is present with the candidate.
<br><br>
For each of the detection, a warning is displayed on the screen like a number of strikes in a baseball game.
<br>Once the candidate crosses a certain number of threshold warnings (here, its set to 3), it displays 'MALPRACTICE DETECTED' and closes the exam window and exits the camera.
