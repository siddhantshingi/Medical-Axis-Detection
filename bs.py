import numpy as np
import cv2 as cv
import math
cap = cv.VideoCapture('./Assignment1/1.mp4')
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# out = cv.VideoWriter('cbsmog1.avi', fourcc, 20.0, (frame_width, frame_height))
# fgbg = cv.createBackgroundSubtractorMOG2(history=100,detectShadows=False)
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
kernel_dilate = np.ones((10,10),np.uint8)
kernel_erosion = np.ones((5,5),np.uint8)
kernel_tophat = np.ones((12,12),np.uint8)
kernal_hitmiss = np.ones((20,20),np.uint8)
scale = 1
delta = 0
ddepth = cv.CV_16S
yavg=0
while(1):
    ret, frame = cap.read()
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    
    fgmask = fgbg.apply(frame)
    
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_TOPHAT, kernel_tophat)
    fgmask = cv.dilate(fgmask,kernel_dilate,iterations = 1)
    # fgmask2 = cv.morphologyEx(fgmask, cv.MORPH_HITMISS, kernal_hitmiss)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel1)
    # fgmask = fgmask - fgmask2
    fgmask = cv.erode(fgmask,kernel_erosion,iterations = 1)

    # fgmask = (255 - fgmask)
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel1)
    
    grad_x = cv.Sobel(fgmask, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(fgmask, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    lines = cv.HoughLinesP(grad,1,np.pi/180,125,0,0)
    prevx1=0 
    prevy1=0
    prevx2=0
    prevy2=0
    minx=10000
    maxx=0
    miny=10000
    maxy=0
    count=0
    slope=0
    if lines is not None:
        x1_min = float("inf")
        x1_max = float("-inf")
        x2_min = float("inf")
        x2_max = float("-inf")
        y1_min = float("inf")
        y1_max = float("-inf")
        y2_min = float("inf")
        y2_max = float("-inf")
        slope_max = float("-inf")
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        minyavg=0
        maxyavg=0
        count_avg = 0
        for l in lines:
            # print l
            x1=l[0][0]
            y1=l[0][1]
            x2=l[0][2]
            y2=l[0][3]
            maxx=max(maxx,x1,x2)
            minx=min(minx,x1,x2)
            maxy=max(maxy,y1,y2)
            miny=min(miny,y1,y2)
            minyavg+=min(y1,y2)
            maxyavg+=max(y1,y2)
            x1_max = max(x1_max,x1)
            x2_max = max(x2_max,x2)
            x1_min = min(x1_min,x1)
            x2_min = min(x2_min,x2)
            y1_max = max(y1_max,y1)
            y2_max = max(y2_max,y2)
            y1_min = min(y1_min,y1)
            y2_min = min(y2_min,y2)
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
            count_avg += 1
            if(x1!=x2):
                slope=slope+((y1-y2)/(x1-x2))
                slope_max = max(slope_max,slope)
                count=count+1
        x1_avg = x1_sum/count_avg
        y1_avg = y1_sum/count_avg  
        x2_avg = x2_sum/count_avg
        y2_avg = y2_sum/count_avg
        minyavg=minyavg/count_avg
        maxyavg=maxyavg/count_avg
        
        if (count != 0):
            slope = slope/count
        if(minx!=10000):
            if(slope>0):
                # cv.line(frame,(maxx,maxy),(minx,miny),(0,255,0),2)
                cv.line(frame,(x1_min,y1_min),(x1_min + 2,y1_min + 2),(255,0,0),5)
                cv.line(frame,(x1_max,y1_max),(x1_max + 2,y1_max + 2),(255,0,0),5)
                cv.line(frame,(x2_min,y2_min),(x2_min + 2,y2_min + 2),(255,0,0),5)
                cv.line(frame,(x2_max,y2_max),(x2_max + 2,y2_max + 2),(255,0,0),5)
                # cv.line(frame,(x1_avg,minyavg),(x2_avg,maxyavg),(0,0,255),7)
                if(yavg==0):
                    yavg=maxyavg
                    cv.line(frame,(x1_avg-(minyavg*(x2_avg-x1_avg)/(maxyavg-minyavg)),0),(x2_avg,maxyavg),(0,0,255),7)
                else:
                    # print yavg," ",maxyavg
                    # if((maxyavg-yavg>50) or (yavg-maxyavg>50)):
                    #     yavg=maxyavg
                    yavg=(yavg+maxyavg)/2
                    cv.line(frame,(x1_avg-(minyavg*(x2_avg-x1_avg)/(maxyavg-minyavg)),0),(x1_avg+((yavg-minyavg)*(x2_avg-x1_avg)/(maxyavg-minyavg)),yavg),(0,0,255),7)
                print yavg," ",maxyavg
            elif(slope<0):
                # cv.line(frame,(minx,maxy),(maxx,miny),(0,255,0),2)
                cv.line(frame,(x1_min,y1_min),(x1_min + 2,y1_min + 2),(255,0,0),5)
                cv.line(frame,(x1_max,y1_max),(x1_max + 2,y1_max + 2),(255,0,0),5)
                cv.line(frame,(x2_min,y2_min),(x2_min + 2,y2_min + 2),(255,0,0),5)
                cv.line(frame,(x2_max,y2_max),(x2_max + 2,y2_max + 2),(255,0,0),5)
                # cv.line(frame,(x1_avg,maxyavg),(x2_avg,minyavg),(0,0,255),7)
                if(yavg==0):
                    yavg=maxyavg
                    cv.line(frame,(x1_avg,maxyavg),(x2_avg-(minyavg*(x1_avg-x2_avg)/(maxyavg-minyavg)),0),(0,0,255),7)
                else:
                    # print yavg," ",maxyavg
                    # if((maxyavg-yavg>50) or (yavg-maxyavg>50)):
                    #     yavg=maxyavg
                    yavg=(yavg+maxyavg)/2
                    cv.line(frame,(x2_avg+((yavg-minyavg)*(x1_avg-x2_avg)/(maxyavg-minyavg)),yavg),(x2_avg-(minyavg*(x1_avg-x2_avg)/(maxyavg-minyavg)),0),(0,0,255),7)
                print yavg," ",maxyavg
            else:
                # cv.line(frame,(((maxx+minx)/2),maxy),(((maxx+minx)/2),miny),(0,255,0),2)
                cv.line(frame,(x1_avg,0),(x2_avg,maxyavg),(0,0,255),7)

    cv.imshow('frame_new',frame)
    # cv.imshow('frame_new',fgmask)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
out.release()
cv.destroyAllWindows()