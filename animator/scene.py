#
# The animation scene, with entities to be captured.
#

import cv2
from copy import deepcopy
import numpy as np

class Scene:
    """ Represents the scene in which the animation takes place. 
    
    Attributes:
        width: Width in pixels of the scene
        height: Height in pixels of the scene
        resolution: Resolution in px/mm of the scene
        entities: Dictionary of Entities on scene, keyed by a string analysis
        entities_init: Copy of entities in their original frame-0 configuration
        background: Background color of the scene.
        origin: Point on image in pixels of the coordinate system origin.
                [int, int] list in the form width, height
        script: Dictonary in the form frame_no : instructions, where frame_no is
                an int and instructions is a list executable python strings.
        current_frame: (int) Current frame number present on scene
        fps: Frames per second of animation. 
    """
    
    def __init__(self, width, height, resolution, fps, entities={}, 
            background=[0, 0, 0]): 
        """ Creates a new scene. 
        Args:
            width: (int) Width in pixels of the scene
            height: (int) Height in pixels of the scene
            resolution: (float) Resolution in px/mm of the scene
            entities: (dict) Dictionary of string : Entity, where key is a 
                      string alias for a particular Entity on the scene.
                      Entities are provided in initial configuration. 
            background: (3-element list) Background color of the scene, bgr
            fps: (int) Frames per second of animation. 
        """
        self.width = width
        self.height = height
        self.resolution = resolution # px/mm
        self.entities = entities
        self.entities_init = deepcopy(entities)
        self.background = background
        self.origin = [width // 2, height // 2]
        self.script = {}
        self.current_frame = 0
        self.fps = fps
        
        self.add_instruction(0, "") # default initialization
    
    def add_entities(self, new_entities):
        """ Adds entities to the scene. Configuration is reflected at frame 0.
        
        Args:
            new_entities: (dict) Dictionary of string : Entity, where key is a 
                          string alias for a particular Entity on the scene.
                          Entities are provided in initial configuration. 
        """
        self.entities.update(new_entities)
        self.entities_init.update(deepcopy(new_entities))
        
    
    def reset(self):
        """ Reverts to frame 0. Overwrites Entities with saved initial values.
        """
        for i in range(len(self.entities)):
            self.entities[i] = deepcopy(self.entities_init[i])
        self.current_frame = 0
    
    def add_instruction(self, frame_no, instruction):
        """ Adds an instruction to the script.
        
        Args:
            frame_no: Frame number in animation to add instruction.
            instruction: (string) A snippet of executable python code. 
                         Most commonly a function call modifying an object.
        """
        if frame_no not in self.script:
            self.script[frame_no] = []
        self.script[frame_no].append(instruction)
    
    # TODO able to play animation backwards?
    def set_frame_no(self, frame_no, display=False): 
        """ Sets scene to the current frame by modifing Entities by the script.
        Plays the scene forward if frame_no > self.current_frame.
        Otherwise, starts from beginning and place up to frame_no.
        
        Args:
            frame_no: (int) Frame number to set the scene to.
            display: (bool) If True, opens a cv2 window at this frame. 
        """
        if frame_no < self.current_frame: # Need to reset to start
            self.reset()
        for i in range(self.current_frame, frame_no):
            if i in self.script: # instructions specified at this step
                for instruction in self.script[i]:
                    try:
                        exec(instruction)
                    except Exception as e:
                        print("FAILED to execute instruction")
                        print("frame no: ", i)
                        print("time: ", i / self.fps)
                        print("instruction: ", '"' + instruction + '"')
                        print()
                        raise e
        self.current_frame = frame_no
        if display:
            cv2.imshow(str(frame_no), img)
            cv2.waitKey(-1)
            cv2.destroyAllWindows()
    
    def capture_frame(self):
        """ Returns an image corresponding to the current state of the scene.

        Returns:
            Shape (height, width, 3) Numpy array; captured image
        """
        
        # Create image & apply background
        img = np.zeros([self.height, self.width, 3], np.uint8) # blank bgr image
        cv2.rectangle(img, (0, 0), (self.width-1, self.height-1),
                self.background, -1) # apply background
        # Capture each entity in scene. 
        # Order in list determines order on screen.
        for alias, entity in self.entities.items():
            entity.add_to_image(img, self.resolution, self.origin)
        return img
        
    def get_frames(self, start_frame=0, end_frame=None, status=True):
        """ Returns a list of frames from playing back the scene.
        
        Args:
            start_frame: Initial frame in outputted list. Default 0 (beginning). 
            end_frame: Final frame in outputted list. Default None (end)
            status (bool): If True, prints completion status to terminal.
            
        Returns:
            list of shape (height, width, 3) frame images 
        """
        if end_frame is None: # Output to end of animation
            end_frame = max(self.script.keys())
        frames = []
        for i in range(start_frame, end_frame + 1):
            if status:
                if i % 10 == 0 or i == end_frame:
                    print("Getting frame no %i of %i" % (i, end_frame))
            self.set_frame_no(i)
            frames.append(self.capture_frame())
        return frames
            
            
        
