# Python program to play a video 
# in reverse mode using opencv  
  
import cv2 
import numpy as np
  
cap = cv2.VideoCapture("./1.mp4") 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((5,5))

check , vid = cap.read() 
counter = 0
check = True
frame_list = [] 

while(check == True): 
    vid = fgbg.apply(vid)
    vid = cv2.dilate(vid,kernel,iterations = 1)
    vid = cv2.morphologyEx(vid, cv2.MORPH_CLOSE, kernel)
    frame_list.append(vid)
    check , vid = cap.read() 
    counter += 1

frame_list.pop() 
  
for frame in frame_list: 
      
    cv2.imshow("Frame" , frame) 
    if cv2.waitKey(25) and 0xFF == ord("q"): 
        break
      
cap.release() 
  
cv2.destroyAllWindows() 