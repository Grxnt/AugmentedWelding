import numpy as np
import cv2
import cv2.aruco as aruco


myDict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

# Create Markers
for i in range(10):
    marker = aruco.drawMarker(myDict, i*10, 600)
     
    cv2.imshow("marker1", marker)
    cv2.waitKey(0)
    path = './markers/image' + str(i) + '.png'
    cv2.imwrite(path, marker)
    
