from entities.dial_indicator import Dial_Indicator
from entities.primatives import *
from entities.dimensions import Leader
import cv2
import numpy as np
import csv
from copy import deepcopy
import funcs
import animation as ani
from scene import Scene
import sys

#
# Temp test animation.
#

    

if __name__ == '__main__':
    
    # # # Setup animation parameters # # #
    width, height = 1920//1, 1080//1
    resolution = 272.//1 # px / mm
    fps = 60
    
    
    # # # Initialize objects # # #
    
    # Create circle # 
    disk = Disk(1.00, color=[200, 50, 50])
    
                          
    # # # Create Scene # # #
    scene = Scene(width, height, resolution, fps)
    scene.add_entities({"disk" : disk})
    
    
    # # # Animate # # #

    ani.pause(scene, 0.5)
    ani.slide(scene, 1, "disk", [1, .5, 0])
    ani.pause(scene, 0.5)
    ani.slide(scene, 1, "disk", [-1, -.5, 0])
    
    ani.pause(scene, 0.5)
    ani.slide_to(scene, 1, "disk", [1, .5, 0])
    ani.pause(scene, 0.5)
    ani.slide_to(scene, 1, "disk", [0, 0, 0])
    ani.pause(scene, 0.5)
    
    
    # # # Write Animation # # #
    
    out = cv2.VideoWriter('temp_ani.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, 
            (width, height))
    frames = scene.get_frames()
    print("\nWriting Animation....")
    for frame in frames:
        out.write(frame)
    out.release()
    print("....done\n")


    print("exiting")
    sys.exit(0)

    
        
