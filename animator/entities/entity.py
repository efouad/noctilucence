#
# entities.entity.py
#
# Contains all entities (Entity objects) that may appear on screen in scenes. 
#


import numpy as np
from pyquaternion import Quaternion
import cv2
from collections.abc import Iterable
import copy


#TODO implement rotation for entities and primatives
class Entity:
    """ A single entity that may be displayed. """
    
    def __init__(self, components=[], opacity=1.0, color=[255, 255, 255], 
            dtype='uint8', size=1, parent=None, pos=[0., 0., 0.], 
            ori=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], visible=True):
        """ Creates a new empty Entity. 
            keyword arguments are formed into a dictionary 'self.attributes', 
            containing all the traits this entity has and stored in the form
            'attribute' : value. 
            Attributes:
                components   -> List of Entities.
                opacity      -> Float from 0-1; 0=transparent, 1=opaque.
                color        -> List [blue, green, red], uint8).
                size         -> Characteristic display thickness in pixels.
                parent       -> Parent Entity containing this Entity. 
                                Pos & orientation is relative to parent csys.
                                parent=None is a global entity, wrt global csys.                                
                pos          -> Origin position of Entity, wrt parent origin.
                                Shape (3) numpy array, or list.
                ori          -> ijk unit vectors axes; orientation wrt parent 
                                Shape (3, 3) numpy array, or list.
                                ijk unit vectors as columns
                visible      -> Boolean determining whether entity is actively
                                visible and should be drawn on the scene
        """
        # Default attributes
        self.attributes = {}

        self.attributes["opacity"] = opacity
        self.attributes["visible"] = visible
        self.attributes["color"] = color # bgr
        self.attributes["size"] = size
        self.attributes["parent"] = parent
        self.attributes["pos"] = np.array(pos)
        self.attributes["ori"] = np.array(ori)
        self.attributes["components"] = [] # List of Entities & points
        self.add_components(components)
    
    def put(self, attribute, new_val):
        """ Sets specified attribute to new value.
        self.put("size", 2) is shorthand for self.attributes["size"] = 2. 
        
        Args:
            attribute (string): key in self.attributes of attribute to change
            new_val (various): new value of attribute to set
        """
        self.attributes[attribute] = new_val
    
    def get(self, attribute):
        """ Gets specified attribute.
        self.get("size") is shorthand for self.attributes["size"]. 
        
        Args:
            attribute (string): Key in self.attributes of attribute to change,
        
        Returns:
            (various) New value of attribute to set.
        """
        return self.attributes[attribute]
    
    def pos(self, copy=False):
        """ Returns this Entity's origin in its parent's csys.
        
        Args:
            copy: If True, returns a copy of origin array, not the array itself
        
        Returns:
            Shape (3) Numpy array; coordinates of the origin of this Entity
        """
        if copy:
            return np.copy(self.attributes["pos"])
        return self.attributes["pos"]
      
    def gpos(self, local_pos=None):
        """ Returns a copy of this Entity's origin wrt global csys.
        
        Recursively navigates through chain of parents until global entity
        (parent=None) is reached.
        
        Args:
            local_pos: Used in recursive calculation. Represents the position
                       that will be reported in the global coordinate system,
                       in the coordinate system of this current Entity. 
                       If local_pos=None, use position of this Entity from pos()
        
        Returns:
            Shape (3) Numpy array; coordinates of the origin of this Entity
        """
        if local_pos is None:  # initial Entity in the recursive stack
            local_pos = self.pos() 
        if self.attributes["parent"] == None:  # end of stack; global parent
            return local_pos
        return self.attributes["parent"].gpos(
                local_pos=self.attributes["parent"].pos() + 
                self.attributes["parent"].ori() @ local_pos)
    
    def ori(self, copy=False):
        """ Returns this Entity's ijk unit vectors wrt parent csys.

        Args:
            copy: If True, returns a copy of the ori array, not the array itself

        Returns:
            Shape (3, 3) Numpy array; ijk unit vectors this Entity as columns
        """
        if copy:
            return np.copy(self.attributes["ori"])
        return self.attributes["ori"]
    
    def gori(self):
        """ Returns a copy of this Entity's ijk unit vectors wrt global csys.
        
        Recursively navigates through chain of parents until global entity
        (parent=None) is reached.
        
        Returns:
            Shape (3, 3) Numpy array; ijk unit vectors this Entity as columns
        """
        
        # Start with the top global orientation, and left multiply the 
        # successive orientation matricies of each child element until we get
        # to self.
        
        if self.attributes["parent"] == None: # global parent
            return np.identity(3)
        return self.ori() @ self.attributes["parent"].gori()
    
    #TODO allow move to gpos?
    def move(self, pos=None, dpos=None, ori=None, dori=None):
         """ Translates and/or rotates this entity within parent csys. 
             
         Args:
             pos: Desired new origin. Shape (3) Numpy array or list.
             dpos: Desired origin delta vector. Shape (3) Numpy array or list.
             ori: Desired new new unit vectors. Shape (3, 3) np array or list. 
             dori: Desired pyquaternion Quaternion to rotate csys. 
         """
         if pos is not None:
             self.attributes["pos"] = np.array(pos)
         elif dpos is not None:
             self.attributes["pos"] += np.array(dpos)
         if ori is not None:
             self.attributes["ori"] = np.array(ori)
         elif dori is not None:
             self.attributes["ori"][:,0] = \
                     dori.rotate(self.attributes["ori"][:,0])
             self.attributes["ori"][:,1] = \
                     dori.rotate(self.attributes["ori"][:,1])
             self.attributes["ori"][:,2] = \
                     dori.rotate(self.attributes["ori"][:,2])
    
    def add_to_image(self, img, resolution, origin):
        """ Adds this entity on the image specified, based on its visibility.
        
        Args:
            img: (3-element Numpy array) The bgr image to be drawn on
            resolution: (float) Pixels per mm in the image
            origin: (2-element list) The xy pixel values of the global origin
        """
        if self.attributes["visible"]:
            if self.attributes["opacity"] == 1:
                self.draw_self(img, resolution, origin)
            elif self.attributes["opacity"] > 0:
                mask = copy.deepcopy(img) # Paint opaque version on here first
                self.draw_self(mask, resolution, origin)
                img[:,:,:] = cv2.addWeighted(img, 1 - self.get_opacity(),
                                      mask, self.get_opacity(), 0)
    
    def draw_self(self, img, resolution, origin):
        """ Draws a representation of this entity on the image specified.
        
        Args:
            img: (3-element Numpy array) The bgr image to be drawn on
            resolution: (float) Pixels per mm in the image
            origin: (2-element list) The xy pixel values of the global origin
        """
        for entity in self.attributes["components"]:
            entity.add_to_image(img, resolution, origin)
            
    def add_components(self, *args):
        """ Adds subcomponent Entities to this Entity. 
            Add any quantity of Entities as arguments, or alternatively add
            lists of Entites as arguments. 
            
        Args:
            *args: Entities to add. Each argument can be a single Entity, or an
                   iterable list of Entities.
        """
        for arg in args:
            if isinstance(arg, Iterable): # argument is a list
                for sub_arg in arg:
                    self.add_components(sub_arg) # break it down recursively
            else:
                if not isinstance(arg, Entity):
                    raise ValueError("%s is not an Entity." % arg)
                self.attributes["components"].append(arg)
                arg.attributes["parent"] = self
                
    def set_attribute(self, attribute, value):
        """ Sets the specified attribute of this Entity to the specified value.
        
        Args:
            attribute: (string) attribute name to set
            value: (type varies) value to set this attribute to.
        """
        self.attributes[attribute] = value
    
    def get_opacity(self):
        """ Returns the total opacity of this entity (product of all parents).
        
        Returns:
            float 0-1: Total opacity of this entity
        """
        if self.attributes["parent"] is None:
            return self.attributes["opacity"]
        return (self.attributes["opacity"] * 
                self.attributes["parent"].get_opacity())
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        