#
# animation.py 
#
# Functions to create animation capture instructions.
# Instructions are executable python strings to modify entities and are
# captured in the script of a Scene object. 
#

from . import funcs
import numpy as np

def pause(scene, duration):
    """ Adds a pause of duration specified to the scene instructions.
    
    Args:
        scene: Scene object whose script will be appended to.
        duration: Time in seconds for event to last. Use -1 for one frame. 
    """
    nframes = max(1, int(duration * scene.fps))
    scene.add_instruction(max(scene.script.keys()) + nframes, "")

def slide(scene, duration, entity_alias, slide_disp=[0, 0, 0], 
        profile="sigmoid", t_start=-1): # TODO Rotation
    """ Moves the entity (relative), and adds motion to scene instructions.
    
    Args:
        scene        (Scene)    Scene object whose script will be appended to.
        duration     (float)    Seconds for event to last. Use -1 for one frame.
        entity_alias (string)   Alias for Entity object whose attribute is to be 
                                changed.
        slide_disp   (list)     3-element list or shape (3) numpy array.
                                Total displacement delta over the duration. 
        profile      (string)   Transition profile shape, see funcs.span.
        t_start      (float)    Time in seconds to begin attribute transition.
                                Use -1 to append to end of last instruction.
    """
    nframes = max(1, int(duration * scene.fps))
    values = funcs.span(np.zeros(3), np.array(slide_disp), nframes, profile)
    
    if t_start == -1: # append to end
        frame_start = max(scene.script.keys())
    else:
        frame_start = int(t_start * scene.fps)
        
    if len(values) == 1:  # e.g. duration is -1; apply at frame start 
        scene.add_instruction(frame_start,
            "self.entities[\"%s\"].move(dpos=%s)" 
            % (entity_alias, list(values[0])))
    else:
        for i in range(1, len(values)):
            scene.add_instruction(frame_start + i,
                "self.entities[\"%s\"].move(dpos=%s)" 
                % (entity_alias, list(values[i] - values[i-1])))

def slide_to(scene, duration, entity_alias, slide_pos=[0, 0, 0], 
        profile="sigmoid", t_start=-1): # TODO Rotation; handle other local csys
    """ Moves the entity (absolute), and adds motion to scene instructions.
    
    Args:
        scene        (Scene)    Scene object whose script will be appended to.
        duration     (float)    Seconds for event to last. Use -1 for one frame.
        entity_alias (string)   Alias for Entity object whose attribute is to be 
                                changed.
        slide_pos   (list)      3-element list or shape (3) numpy array.
                                Position for entity to move to, in its csys. 
        profile      (string)   Transition profile shape, see funcs.span.
        t_start      (float)    Time in seconds to begin attribute transition.
                                Use -1 to append to end of last instruction.
    """
    nframes = max(1, int(duration * scene.fps))
    values = funcs.span(0, 1, nframes, profile)
    
    if t_start == -1: # append to end
        frame_start = max(scene.script.keys())
    else:
        frame_start = int(t_start * scene.fps)
        
    if len(values) == 1:  # e.g. duration is -1; apply at frame start 
        scene.add_instruction(frame_start,
            "self.entities[\"%s\"].move(pos=%s)" 
            % (entity_alias, list(slide_pos)))
    else:
        for i in range(1, len(values)):
            scene.add_instruction(frame_start + i, (
                "self.entities[\"%s\"].move(dpos=(np.array(%s) - " +  
                "self.entities[\"%s\"].pos()) * (%s))")
                % (entity_alias, list(slide_pos), entity_alias, 
                (values[i] - values[i-1]) / (values[-1] - values[i-1])))

def sweep_cmd(scene, duration, cmd, t_start=-1):
    """ Repeats a command as an instruction for each frame in a given interval.
    
    Args:
        scene        (Scene)    Scene object whose script will be appended to.
        duration     (float)    Seconds for event to last. Use -1 for one frame.
        cmd          (string)   String of command for scene to execute.
        t_start      (float)    Time in seconds to begin attribute transition.
                                Use -1 to append to end of last instruction.
    """
    nframes = max(1, int(duration * scene.fps))
    if t_start == -1: # append to end
        frame_start = max(scene.script.keys())
    else:
        frame_start = int(t_start * scene.fps)
    for i in range(nframes):
        scene.add_instruction(frame_start + i, cmd)

def set_cmd(scene, cmd, t_start=-1):
    """ Adds a command at the time interval specified.
    
    Args:
        scene        (Scene)    Scene object whose script will be appended to.
        cmd          (string)   String of command for scene to execute.
        t_start      (float)    Time in seconds to begin attribute transition.
                                Use -1 to append to end of last instruction.
    """
    sweep_cmd(scene, -1, cmd, t_start=t_start)

def sweep_attr(scene, duration, entity_alias, attribute, start, end, 
        profile="sigmoid", t_start=-1):
    """ Sweeps the attribute of entity, and adds to scene instructions.
    
    Args:
        scene        (Scene)    Scene object whose script will be appended to.
        duration     (float)    Seconds for event to last. Use -1 for one frame.
        entity_alias (string)   Alias for Entity object whose attribute is to be 
                                changed.
        attribute    (string)   Attribute name to change.
        start        (varies)   Start value of attribute. 
        end          (varies)   End value of attribute.
        profile      (string)   Transition profile shape, see funcs.span.
        t_start      (float)    Time in seconds to begin attribute transition.
                                Use -1 to append to end of last instruction.
    """
    nframes = max(1, int(duration * scene.fps))
    values = funcs.span(start, end, nframes, profile)
    
    if t_start == -1: # append to end
        frame_start = max(scene.script.keys())
    else:
        frame_start = int(t_start * scene.fps)
        
    for i in range(len(values)):
        scene.add_instruction(frame_start + i,
            "self.entities[\"%s\"].set_attribute(\"%s\", %f)"
            % (entity_alias, attribute, values[i]))

def set_attr(scene, entity_alias, attribute, value, t_start=-1, **kwargs):
    """ Sets the attribute of an entity to a value at the time specified.
    See sweep_attr.
    """
    sweep_attr(scene, -1, entity_alias, attribute, 0, value, **kwargs)
    

def fade_in(scene, duration, entity_alias, t_start=-1, **kwargs):
    """ Fades specified entities to max opacity, and adds to scene instructions. 
    Sets visibility to True.
    See animation.sweep_attr.
    """
    # Find original opacity: Existing if already visible; 0 if not yet visible. 
    set_attr(scene, entity_alias, "visible", True, t_start=t_start, **kwargs)
    sweep_attr(scene, duration, entity_alias, "opacity", 0, 1, t_start=t_start, 
            **kwargs)
    

def fade_out(scene, duration, entity_alias, t_start=-1, **kwargs):
    """ Fades specified entities to min opacity, and adds to scene instructions. 
    Sets visibility to False. 
    See animation.sweep_attr.
    """
    sweep_attr(scene, duration, entity_alias, "opacity", 1, 0, t_start=t_start, 
            **kwargs)
    set_attr(scene, entity_alias, "visible", False, t_start=t_start, **kwargs)
    


    
    
    
    