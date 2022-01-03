#
# Represents an active agent that participates in the animation. 
#

from entities import *

class Actor:
    """ Represents an active agent that participates in the animation. 
        An actor is responsible for one Entity, and contains a script of 
        time-stamped instructions for how that Entity will be modified 
        throughout the animation. """
    
    def __init__(self, entity=Entity(), n_frames=1, script={}, 
            script_attributes=[]):
        """ Creates a new actor, consisting of its geometric form 'entity', and
            a list 'script_attributes' consisting of all keys (str) that will be 
            added to self.script. self.script is initialized to 'script', a 
            dictionary with contents of the form 'attribute' : np.array, where 
            np.array is a 1-dimensional array of length 'n_frames' corresponding
            to the value of this attribute in each animation frame. """ 
            #TODO consider scipy sparse list instead. 
        self.entity = entity
        self.script = {}
        for script_attribute in script_attributes:
            if script_attribute == "color":
                self.script[script_attribute] = \
                        -1 * np.ones([n_frames, 3], dtype='int16')
            else:
                self.script[script_attribute] = \
                        -1 * np.ones([n_frames, 1], dtype='float64')
            # star character signifies no change specified at this frame. 
        
    def act(self, frame_no):
        """ Actor modifies its entity into the desired form at frame timestep
            'frame_no', according to the rules of its script. """
        for attribute in self.script:
            if frame_no in range(np.size(self.script[attribute])):
                if not -1 in self.script[attribute][frame_no]: 
                    # Value is specified at this frame:
                    self.entity.attributes[attribute] = \
                            self.script[attribute][frame_no]

    def fade(self, start_frame, end_frame, start_val, end_val, 
            profile="linear"):
        """ Adds an opacity modifier to the script. Opacity changes from
            'start_val' to 'end_val' (from 0-1) over frame range 'start_frame' 
            to 'end_frame', according to 'profile' specified. 
            See span for profile options. """
        span_vals = start_val + (end_val - start_val) * \
                self.span(end_frame - start_frame + 1, profile)
        for i in range(end_frame - start_frame + 1):
            self.script["opacity"][i + start_frame][0] = span_vals[i]
            
    def color_shift(self, start_frame, end_frame, start_val, end_val, 
            profile="linear"):
        """ Adds a color shift modifier to the script. Color changes from
            'start_val' (list) to 'end_val' (list) over the frame range 
            'start_frame' to 'end_frame', per the profile specified. """
        span_vals = len(start_val) * [0]
        for i in range(len(start_val)):
            span_vals[i] = start_val[i] + (end_val[i] - start_val[i]) * \
                    self.span(end_frame - start_frame + 1, profile)
            for j in range(end_frame - start_frame + 1):
                self.script["color"][j + start_frame][i] = span_vals[i][j]
    
