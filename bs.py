import numpy as np
import cv2 as cv
cap = cv.VideoCapture('./Assignment1/1.mp4')
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# out = cv.VideoWriter('cbsmog1.avi', fourcc, 20.0, (frame_width, frame_height))
# fgbg = cv.createBackgroundSubtractorMOG2(history=100,detectShadows=False)
fgbg=cv.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((10,2),np.uint8)
scale = 1
delta = 0
ddepth = cv.CV_16S
while(1):
    ret, frame = cap.read()
    frame = cv.GaussianBlur(frame, (3, 3), 0)
    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray,50,150,apertureSize = 3)
    fgmask1 = fgbg.apply(frame)
    # out.write(fgmask)
    fgmask = cv.dilate(fgmask1,kernel,iterations = 1)
    # fgmask=cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    grad_x = cv.Sobel(fgmask, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # grad_x = cv.Scharr(fgmask, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(fgmask, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # grad_y = cv.Scharr(fgmask, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    minLineLength = 100
    maxLineGap = 10
    lines = cv.HoughLinesP(grad,1,np.pi/180,125,0,0)
    minx=10000
    maxx=0
    miny=10000
    maxy=0
    count=0
    slope=0
    # print lines
    if lines is not None:
        for l in lines:
            x1=l[0][0]
            y1=l[0][1]
            x2=l[0][2]
            y2=l[0][3]
            maxx=max(maxx,x1,x2)
            minx=min(minx,x1,x2)
            maxy=max(maxy,y1,y2)
            miny=min(miny,y1,y2)
            if(x1!=x2):
                slope=slope+((y1-y2)/(x1-x2))
                count=count+1
    if(minx!=10000):
        # if(count>0):
        #     avgslope=slope/count
        #     delta=int((miny-maxy)/(2.0*avgslope))
        #     cv.line(frame,(((maxx+minx)/2)-delta,maxy),(((maxx+minx)/2)+delta,miny),(0,255,0),2)
        # else:
        if(slope>0):
            cv.line(frame,(maxx,maxy),(minx,miny),(0,255,0),2)
        elif(slope<0):
            cv.line(frame,(minx,maxy),(maxx,miny),(0,255,0),2)
        else:
            cv.line(frame,(((maxx+minx)/2),maxy),(((maxx+minx)/2),miny),(0,255,0),2)
    # for l in lines:
    #     rho=l[0][0]
    #     theta=l[0][1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a*rho
    #     y0 = b*rho
    #     x1 = int(x0 + 1000*(-b))
    #     y1 = int(y0 + 1000*(a))
    #     x2 = int(x0 - 1000*(-b))
    #     y2 = int(y0 - 1000*(a))

    #     cv.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

    # cv.imwrite('houghlines3.jpg',img)
    cv.imshow('frame',frame)

    # frame = cv.flip(frame, 0)
    # write the flipped frame
    # out.write(frame)
    # cv.imshow('frame', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
out.release()
cv.destroyAllWindows()
