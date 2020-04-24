# People_Detection_Tracking-CPU
![Test Image 4](https://github.com/vyzboy92/people_detection_tracking-CPU/blob/master/Video/demo.png)

This repo lets you setup a quick demo showcasing people detection, tracking and counting, all performed on your CPU.
OpenVINO toolkit enables CPU processing and you can have this code demonstrated on any PC having Core i5 6th Gen or higher.
 
 ## Reqirements
 1. OpenCV
 2. OpenVINO
 3. imutils
 
 ## Usage
 Start by editing the ```config.json``` file. Change value of the key ```file``` in json to the path of your video, rtsp stream from IP camera or your webcam connected to PC ( note: webcams id needs to be specified as integer - 0, 1 etc)
 
 Run ```python multi_camera_people_detection.py```
