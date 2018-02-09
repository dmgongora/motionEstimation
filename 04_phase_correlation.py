import sys
import numpy as np
import cv2


def getROI(frame, size, xy, show = False):
    '''
    Parameters:
    - frame: single video frame in colors (3D matrix)
    - size: ROI size (square shaped region)
    - xy: coordinates of the upper left corner of the ROI
    - show: show ROI (boolean)
    '''

    pt1 = (xy[0], xy[1])
    pt2 = (xy[0] + size, xy[1] + size)

    roi = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    if show:
        cv2.rectangle(frame, pt1, pt2, (0,255,0), thickness = 5)

    return roi

if __name__ == '__main__':
    filename = "./../terrainVibrations/{}".format(sys.argv[1])
    print("Loading "+filename)
    cap = cv2.VideoCapture(filename)

    # Take first frame 
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width, frame_ch = old_frame.shape

    
    ROI_SIZE = frame_height - 10
    roi_coord = (int(0.5*frame_width - 0.5*ROI_SIZE), 10)    # Upper left corner
    roi_old = getROI(old_gray, ROI_SIZE, roi_coord)

    # Create 2D filter to remove possible edge effects
    win_width, win_height = roi_old.shape
    win = cv2.createHanningWindow((win_height, win_width), cv2.CV_32F)

    pt0 = (int(0.5*frame_width), int(0.5*frame_height)-250) # at the center of the frame 

    while(1):
        ret, frame = cap.read()
        if (ret == False):
            break 

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = getROI(frame_gray, ROI_SIZE, roi_coord, True)

        # Phase correlation
        ret = cv2.phaseCorrelate(np.float32(roi_old), np.float32(roi), win)
        # Correlation, Displacement in X, Displacement in Y
        print ret[1], ret[0][0], ret[0][1]

        # Draw direction
        cv2.line(frame_gray, pt0, (pt0[0]-int(5*ret[0][0]), pt0[1]-int(5*ret[0][1])),
                 (0,255,0), thickness = 4)
        
        #cv2.imshow('FPV video', frame_gray)
        cv2.waitKey(1)
        # Now update the previous frame and previous points
        roi_old = roi.copy()
        #old_gray = frame_gray.copy()

    cv2.destroyAllWindows()
    cap.release()
