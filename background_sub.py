# Python program to play a video 
# in reverse mode using opencv  
  
import cv2 
import numpy as np
  
cap = cv2.VideoCapture("./1.mp4") 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

  
check , vid = cap.read() 
counter = 0
check = True
frame_list = [] 

while(check == True): 
    vid_fgmask = fgbg.apply(vid)
    frame_list.append(vid_fgmask)
    # cv2.imwrite("frame%d.jpg" %counter , vid_fgmask)
    check , vid = cap.read() 
    counter += 1

frame_list.pop() 
  
for frame in frame_list: 
      
    cv2.imshow("Frame" , frame) 
    if cv2.waitKey(25) and 0xFF == ord("q"): 
        break
      
cap.release() 
  
cv2.destroyAllWindows() 