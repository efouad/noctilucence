#
# Sandbox
#

import cv2
import numpy as np
import entities


if __name__ == '__main__':
    img = np.zeros([1000, 1500, 3], np.uint8) # blank bgr image
    
    ##cv2.circle(img, (200, 300), 5, (255, 255, 255), cv2.FILLED, lineType=cv2.LINE_AA) 
    ##cv2.line(img, (-100, -125), (3000, 4000), (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    
    p1 = entities.Point([10, 20, 0], size=5)
    p2 = entities.Point([-3, -9.2, 0], size=5)
    p3 = entities.Point([-20, 4, 0], size=5)
    
    l1 = entities.Line([0, 0, 0], [.707, .707, 0], size=1)
    
    resolution = 10
    origin = [270, 480]
    
    p1.draw_self(img, resolution, origin)
    p2.draw_self(img, resolution, origin)
    p3.draw_self(img, resolution, origin)
    l1.draw_self(img, resolution, origin)
    
    
    cv2.imshow('image',img)
    cv2.waitKey(0)