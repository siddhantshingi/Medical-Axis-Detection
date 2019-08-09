# Python program to play a video 
# in reverse mode using opencv  
  
import cv2 
  
cap = cv2.VideoCapture("./1.mp4") 
  
check , vid = cap.read() 
counter = 0
check = True
frame_list = [] 

while(check == True): 
    check , vid = cap.read() 
    frame_list.append(vid) 
    counter += 1

frame_list.pop() 
  
for frame in frame_list: 
      
    cv2.imshow("Frame" , frame) 
    if cv2.waitKey(25) and 0xFF == ord("q"): 
        break
      
cap.release() 
  
cv2.destroyAllWindows() 

frame_list.reverse() 
for frame in frame_list: 
    cv2.imshow("Frame" , frame) 
    if cv2.waitKey(25) and 0xFF == ord("q"): 
        break
  
cap.release() 
cv2.destroyAllWindows() 