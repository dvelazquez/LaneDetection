# Daniel Velazquez
# May 2017
# Based in many of the Udacity's Self Driving Car Nano Degree Students
# http://davidlichtenberg.com/articles/2017-03/lane-detection
# Others

import numpy as np
import cv2

def nothing():
    pass

cap = cv2.VideoCapture(1)

low_threshold = 100
high_threshold = 200
minLineLength = 30
maxLineGap = 10
rho = 1
theta = 180


cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN);
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

cv2.namedWindow('Tracker')

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    cv2.createTrackbar('Lo Thresh', 'Tracker', 0, 255, nothing)
    low_threshold = cv2.getTrackbarPos('Lo Thresh', 'Tracker')
    cv2.createTrackbar('Hi Thresh', 'Tracker', 0, 255, nothing)
    high_threshold = cv2.getTrackbarPos('Hi Thresh', 'Tracker')
    cv2.createTrackbar('Min Line', 'Tracker', 0, 255, nothing)
    minLineLength = cv2.getTrackbarPos('Min Line', 'Tracker')
    cv2.createTrackbar('Max Line', 'Tracker', 0, 50, nothing)
    maxLineGap = cv2.getTrackbarPos('Max Line', 'Tracker')
    cv2.createTrackbar('rho', 'Tracker', 1, 100, nothing)
    rho = cv2.getTrackbarPos('rho', 'Tracker')
#    cv2.createTrackbar('theta', 'Tracker', 1, 360, nothing)
#    theta = cv2.getTrackbarPos('theta', 'Tracker')



    frame=img
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img, low_threshold, high_threshold)

    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    # screen is 960W, 540H
    vertices = np.array( [[[250,250],[50,550],[950,550],[750,250]]], dtype=np.int32 )
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    thickness = 15
    lines = cv2.HoughLinesP(img,rho,np.pi/theta,thickness,minLineLength,maxLineGap)

    # def draw_lines
    imshape = img.shape
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = [] 
    y_min = img.shape[0]
    y_max = int(img.shape[0]*0.611)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)) < 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                left_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                left_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#           cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
            elif ((y2-y1)/(x2-x1)) > 0:
                mc = np.polyfit([x1, x2], [y1, y2], 1)
                right_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                right_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#           cv2.line(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
    print "Lo Threshold %d" % low_threshold
    print "Hi Threshold %d" % high_threshold
    print "Mine Line Length %d" % minLineLength
    print "Max Line Gap %d" % maxLineGap

    # Need to fill out the empty lines with anything, something better than ones
    if len(left_x1) == 0:
        left_x1 = [1]
    if len(left_x2) == 0:
        left_x2 = [1]
    if len(right_x1) == 0:
        right_x1 = [1]
    if len(right_x2) == 0:
        right_x2 = [1]

    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(np.nanmean(right_x1))
    r_avg_x2 = np.int(np.nanmean(right_x2))
#     print([l_avg_x1, l_avg_x2, r_avg_x1, r_avg_x2])

    cv2.line(frame, (l_avg_x1, y_min), (l_avg_x2, y_max), [0, 0, 255], thickness)
    cv2.line(frame, (r_avg_x1, y_min), (r_avg_x2, y_max), [0, 0, 255], thickness)    


#    cv2.imshow("Camera",frame)
#    cv2.imshow('Tracker', masked_image)
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
