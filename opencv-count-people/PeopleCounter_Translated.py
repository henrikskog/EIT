## Contrast based People Counter with OpenCV
## Original code by Federico Mejia
## Modified by: Chaney Saetre for NTNU Course: TDT4860 - Digitale tvillinger

import numpy as np
import cv2 as cv
import Person
from time import strftime, time
from sys import exit
from os import getcwd

try:
    log = open('log.txt', "w")
    print("Log file opened successfully")
except:
    print("Cannot open log file")
    exit(1)

print(f"Current working directory: {getcwd()}")

# Counters
cnt_up   = 0
cnt_down = 0

# Video Source
esp32_stream_url = "http://192.168.1.191:81/stream"
cap = cv.VideoCapture(esp32_stream_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not connect to ESP32-CAM stream")
    print("Please check if the ESP32-CAM is powered on and connected to the network")
    print(f"Make sure the stream URL is correct: {esp32_stream_url}")
    exit(1)

print("Connected to ESP32-CAM stream successfully")

# Prints capture properties to console
for i in range(19):
    print( i, cap.get(i))

h = 480
w = 640
frameArea = h*w
areaTH = frameArea/100              # Optimized for 640x480 frame size wide angle lens at 2.4m height
print(f"Area Threshold {areaTH}")   # Movement group at atleast 1% of the frame area counts as a moving person

# Boundary lines
line_up = int(5*(h/10))
line_down = int(6*(h/10))

up_limit = int(1*(h/5))
down_limit = int(4*(h/5))

print(f"Red line y: {line_down}")
print(f"Blue line y: {line_up}")
line_down_color = (255,0,0)
line_up_color = (0,0,255)

pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 =  [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

# Background Subtractor
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)

# Morphological filters
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variables
font = cv.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 150                         # max age of a person in number of frames
pid = 1

# Variables to store total processing time and frame count
total_processing_time = 0
frame_count = 0

while(cap.isOpened()):
    start_time = time()  # Start time for processing the frame
    # Read frame from video stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from ESP32-CAM stream")
        print("Attempting to reconnect...")
        cap = cv.VideoCapture(esp32_stream_url)
        continue

    for i in persons:
        i.age_one()                     #age every person one frame
    ######################
    #   PRE-PROCESSING   #
    ######################
    
    # Apply backgroun subtraction
    fgmask = fgbg.apply(frame)
    
    # Binaryization of image
    try:
        # Forcing image to be B&W, pixel with intensity > 200 will be white and the rest black
        ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)            
        # Opening (erode->dilate) to remove noise.
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        # Closing (dilate -> erode) to join white regions.
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
    except:
        print("End Of Function")
        print(f"UP: {cnt_up}")
        print (f"DOWN: {cnt_down}")
        break
    ################
    #   CONTOURS   #
    ################
    
    # RETR_EXTERNAL returns only extreme outer flags. All child contours are left behind.
    contours0, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if area > areaTH:
            ################
            #   TRACKING   #
            ################
            
            # Need to add conditions for multi-person, screen inputs and outputs.
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)

            new = True
            if cy in range(up_limit, down_limit):
                for i in persons:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        # The object is close to one that was already detected before
                        new = False
                        i.updateCoords(cx,cy)   # Updates coordinates on the object and resets age
                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1
                            print(f"ID:{i.getId()} going up at {strftime('%c')}")
                            log.write(f"ID:{i.getId()} going up at {strftime('%c')} \n")
                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1
                            print(f"ID:{i.getId()} going down at {strftime("%c")}")
                            log.write(f"ID:{i.getId()} going down at {strftime("%c")} \n")
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        # Remove i from the persons list
                        index = persons.index(i)
                        persons.pop(index)
                        del i     # Free the memory of i
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     
            ##########################
            #   COUNTOUR FRAMING     #
            ##########################
            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
            cv.drawContours(frame, cnt, -1, (0,255,0), 3)
            
    ###########################
    #   TRAJECTORY PLOTTING   #
    ###########################
    for i in persons:
        if len(i.getTracks()) >= 2:
            pts = np.array(i.getTracks(), np.int32)
            pts = pts.reshape((-1,1,2))
            frame = cv.polylines(frame,[pts],False,i.getRGB())
        if i.getId() == 9:
            print (str(i.getX()), ',', str(i.getY()))
        cv.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv.LINE_AA)
        
    ####################
    #   VIDEO STREAM   #
    ####################
    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)
    frame = cv.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    cv.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv.LINE_AA)
    cv.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv.LINE_AA)

    cv.imshow('Frame',frame)
    cv.imshow('Mask',mask)    

    # Press ESC to quit program
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break

    end_time = time()  # End time for processing the frame
    total_processing_time += end_time - start_time
    frame_count += 1

# Calculate average processing time per frame and average FPS
average_processing_time = total_processing_time / frame_count if frame_count > 0 else 0
average_fps = 1 / average_processing_time if average_processing_time > 0 else 0

print(f"Average FPS: {average_fps:.2f}")
log.write(f"Average FPS: {average_fps:.2f} \n")

print(f"Average processing time per frame: {average_processing_time:.4f} seconds")
log.write(f"Average processing time per frame: {average_processing_time:.4f} seconds \n")

################
#   SHUTDOWN   #
################
print("Flushing and closing log file")
log.flush()
log.close()
print("Log file closed successfully")
cap.release()
cv.destroyAllWindows()