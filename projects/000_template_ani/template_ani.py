#
# template_ani.py
#
# A template test animation. 
#



# Add top level package to path
import sys
import os
from os import path
local_path = os.path.abspath(__file__)
pack_name = "noctilucence"
pack_path = local_path[:local_path.index(pack_name) + len(pack_name)]
sys.path.insert(0, pack_path)

import cv2
import numpy as np

import animator
from animator import animation as ani
from animator.scene import Scene
from animator import funcs

from animator.entities.primatives import Disk
    


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
    
    out = cv2.VideoWriter('template_ani.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frames = scene.get_frames()
    print("\nWriting Animation....")
    for frame in frames:
        out.write(frame)
    out.release()
    print("....done\n")

    print("exiting")
    sys.exit(0)

    
        
