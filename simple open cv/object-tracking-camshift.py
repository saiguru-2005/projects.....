# DataFlair Object Tracker

# import necessary packages
import cv2
import numpy as np

# Naming the Output window
windowname = 'Result'
cv2.namedWindow(windowname)

cap = cv2.VideoCapture(0)

output = None

x, y, w, h = 0, 0, 0, 0

first_point_saved = False
second_point_saved = False
track_window = (x, y, w, h)
can_track = False

def click_event(event, px, py, flags, param):
    global x, y, w, h, first_point_saved,second_point_saved, track_window, can_track, output

    # Left mouse button release event
    if event == cv2.EVENT_LBUTTONUP:
        if first_point_saved:
            w = px-x
            h = py-y
            track_window = (x, y, w, h)
            print(x, y, w, h)
            first_point_saved = False
            second_point_saved = True
        else:
            x = px
            y = py
            first_point_saved = True
            can_track = False
            

    # Right mouse button press event
    if event == cv2.EVENT_RBUTTONDOWN:
        can_track = False
cv2.setMouseCallback(windowname, click_event)  # Start the mouse event



# initialize tracker 

def initialize(frame, track_window):
    x, y, w, h = track_window
    # set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    roi_hist = cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist, roi

# Setup the termination criteria
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check if 2nd point is also saved then initialize the tracker
    if second_point_saved:
        roi_hist, roi = initialize(frame, track_window)
        second_point_saved = False
        can_track = True
    
    # Start tracking
    if can_track == True:
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        print(ret)
        print(track_window)
        cv2.imshow('roi', roi)
        output = cv2.polylines(frame,[pts],True, 255,2)
        
    else:
        output = frame
        if first_point_saved:
            cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
        cv2.destroyWindow('roi')
        

    # Show the output
    cv2.imshow(windowname,output)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()