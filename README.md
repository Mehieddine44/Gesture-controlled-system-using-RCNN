# Gesture-controlled-system-using-RCNN

- This project is an attempt to control operating systems using convolutional neural networks.
To acheive this goal, I used Tensorflows's object detection API which provides pre trained models from which you can 
select one to perform transfer learning on. I trained an RCNN model to detect human fingers in a live video feed 
from the front camera of my device. Then mapped the coordinates of the detected finger to control the mouse on my
windows system.
- This is a demonstration of the result obtained :

![alt text](https://github.com/Mehieddine44/Gesture-controlled-system-using-RCNN/blob/master/result%201.PNG)
![alt text](https://github.com/Mehieddine44/Gesture-controlled-system-using-RCNN/blob/master/result%202.PNG)
![alt text](https://github.com/Mehieddine44/Gesture-controlled-system-using-RCNN/blob/master/result%203.PNG)
![alt text](https://github.com/Mehieddine44/Gesture-controlled-system-using-RCNN/blob/master/result%204..PNG)

- in this repository I only provide the final model and the scripts needed to use it. If you want to train a similar model, you 
should follow this great tutorial by EdjeElectronics :

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# How to use it :

- install : Python + Tensorflow (preferably GPU version) + opencv + pyautogui (to be able to move mouse).
- download the zip file of the repository.
- download the RCNN model from here : https://drive.google.com/open?id=14RKWjA02cGwntYAwPm42Sw0RltBGzCwh
- move the downloaded .pb file to the master folder.
- run the object_detection_webcam.py script.
