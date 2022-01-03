#
# Coordinate systems contained by entities, cameras, and stages. 
#

import numpy as np
from pyquaternion import Quaternion

class CSys:
    """ Represents a single coordinate system. """
    
    def __init__(self, origin=np.array([0, 0, 0]), 
            vecs=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],dtype='float64').T, 
            parent=None):
        """ Creates a new coordinate system with respect to the 'parent' 
            CSys given. If 'parent'==None, use absolute coordinates. 'origin' 
            is a 1d 3-element np array (mm). 'vecs' are the unit vectors of the 
            coordinate system as column vectors. """
        self.origin = origin
        self.vectors = vecs
        self.parent_csys = parent
        
    def parent(self, new_parent=None):
        """ Returns the parent coordinate system as a CSys.  
            If 'new_parent' is not None, sets the parent to 'new_parent'
            before returning it. """
        if new_parent is not None:
            self.parent_csys = new_parent
        return self.parent_csys
        
    def o(self, new_origin=None):
        """ Returns the origin of the coordinate system as a 1-d 3-element 
            np.array. If 'new_origin' is not None, sets the origin as a deep
            copy of 'new_origin' before returning it. """
        if new_origin is not None:
            self.origin = np.copy(new_origin)
        return self.origin

    def vecs(self, new_vecs=None):
        """ Returns the unit vectors of the coordinate system as a 2-d 3x3 
            np.array. If 'new_vecs' is not None, sets the vectors as a deep
            copy of 'new_vecs' before returning it. """
        if new_vecs is not None:
            self.vectors = np.copy(new_vecs)
        return self.vectors
    
    def x(self, new_x=None):
        """ Returns the x column vector of self.vecs. 
            if 'new_x' is not None, sets the x column vector as a deep copy
            before returning it. """
        if new_x is not None:
            self.vectors[:, 0] = new_x
        return self.vectors[:, 0]
        
    def y(self, new_y=None):
        """ Returns the y column vector of self.vecs. 
            if 'new_y' is not None, sets the y column vector as a deep copy
            before returning it. """
        if new_y is not None:
            self.vectors[:, 1] = new_y
        return self.vectors[:, 1]
        
    def z(self, new_z=None):
        """ Returns the z column vector of self.vecs. 
            if 'new_z' is not None, sets the z column vector as a deep copy
            before returning it. """
        if new_z is not None:
            self.vectors[:, 2] = new_z
        return self.vectors[:, 2]
        
    def move(self, new_pos=None, delta_pos=None, new_vecs=None, 
            delta_quaternion=None):
         """ Modifies the position and orientation of this coordinate system.
             'new_pos' is the desired new origin as a shape (3) np.array.
             'delta_pos' is the origin adjust amount as a shape (3) np.array. 
             'new_vecs' are new unit vectors as a shape (3, 3) np.array. 
             'delta_quaternion' is a quaternion to rotate the CSys. """
         if new_pos is not None:
             self.o(new_pos)
         elif delta_pos is not None:
             self.o(self.origin + delta_pos)
         if new_vecs is not None:
             self.vecs(new_vecs)
         elif delta_quaternion is not None:
             self.x(delta_quaternion.rotate(self.x()))
             self.y(delta_quaternion.rotate(self.y()))
             self.z(delta_quaternion.rotate(self.z()))
             
             
