#
# leader_test.py
#
# Leader line animation testing
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

from animator.entities.dimensions import Leader



if __name__ == "__main__":

    # # # Setup animation parameters # # #
    width, height = 1920//1, 1080//1
    resolution = 272.//1 # px / mm
    fps = 60

    
    # # # Initialize objects # # #
    lead0 = Leader([], 
                   size=2, color=[0, 0, 255])

    lead1 = Leader([[-1, -.5, 0]], 
                   size=1, color=[0, 255, 0])
                   
    lead2 = Leader([[-1+1, -.5+1, 0], 
                    [.7+1.5, .5+.5, 0]], 
                   size=5, color=[255, 255, 0], 
                   start_arrow=True)
                   
    lead3 = Leader([[-1-.5, -.5-.3, 0], 
                    [.7-.9, .5+1, 0], 
                    [1.-1.1, .5+.3, 0]], 
                   size=2, color=[255, 255, 0],
                   end_arrow=True)

    lead4 = Leader([[1, -.5, 0], 
                    [-.7, .5, 0], 
                    [1., 0, 0], 
                    [1., .5, 0]], 
                   size=2, color=[255, 0, 255],
                   start_arrow=False, end_arrow=False)

    lead5 = Leader([[-1, -.5, 0], 
                    [.7, .5, 0], 
                    [1., .5, 0], 
                    [1., -.5, 0],
                    [.1, -.1, 0]], 
                   size=3, color=[100, 155, 200], 
                   start_arrow=True, end_arrow=True)
    
    # # # Create Scene # # #
    scene = Scene(width, height, resolution, fps)
    scene.add_entities({"lead5" : lead5, "lead4" : lead4, "lead3" : lead3,
            "lead2" : lead2, "lead1" : lead1, "lead0" : lead0})

    
    # # # Animate # # #
    ani.pause(scene, 1.0)
    
    ani.sweep_attr(scene, 1.0, "lead5", "extension", 0.0, 1.0, 
        profile="sinusoid", t_start=-1)
    ani.pause(scene, 1.0)
    
    ani.sweep_attr(scene, 1.0, "lead4", "extension", 0.0, 1.0, 
        profile="sinusoid", t_start=-1)
    ani.pause(scene, 1.0)
    
    ani.sweep_attr(scene, 1.0, "lead3", "extension", 0.0, 1.0, 
        profile="sinusoid", t_start=-1)
    ani.pause(scene, 1.0)
    
    ani.sweep_attr(scene, 1.0, "lead2", "extension", 0.0, 1.0, 
        profile="sinusoid", t_start=-1)
    ani.pause(scene, 1.0)
    
    # # ani.sweep_attr(scene, 1.0, "lead1", "extension", 0.0, 1.0, 
        # # profile="sinusoid", t_start=-1)
    # # ani.pause(scene, 1.0)
    
    # # ani.sweep_attr(scene, 1.0, "lead0", "extension", 0.0, 1.0, 
        # # profile="sinusoid", t_start=-1)
    # # ani.pause(scene, 1.0)
  
  
    # # # Write Animation # # #
    out = cv2.VideoWriter('leader_test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, (width, height))
    frames = scene.get_frames(status=False)
    print("\nWriting Animation....")
    for frame in frames:
        out.write(frame)
    out.release()
    print("....done\n")