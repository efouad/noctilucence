#
# Contains all entities that may appear in scenes on the screen. 
#

import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion
import cv2
from collections.abc import Iterable





### A generic entity. An object that may appear on the screen. ###

class Entity:
    """ A single entity that may be displayed. """
    
    def __init__(self, components=[], opacity=1.0, color=[255, 255, 255], 
            dtype='uint8', size=1, parent=None, pos=[0., 0., 0.], 
            ori=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]):
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
        """
        # Default attributes
        self.attributes = {}
        self.attributes["opacity"] = opacity
        self.attributes["color"] = color # bgr
        self.attributes["size"] = size
        self.attributes["parent"] = parent
        self.attributes["pos"] = np.array(pos)
        self.attributes["ori"] = np.array(ori)
        self.attributes["components"] = [] # List of Entities & points
        self.add_components(components)
    
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
    
    def draw_self(self, img, resolution, origin):
        """ Draws this entity on the image specified.
        
        Args:
            img: (3-element Numpy array) The bgr image to be drawn on
            resolution: (float) Pixels per mm in the image
            origin: (2-element list) The xy pixel values of the global origin
        """
        for entity in self.attributes["components"]:
            entity.draw_self(img, resolution, origin)
            
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




### Primitive geometric Entities. ###

class Point(Entity):
    """ A single 3d point. 
        
        Attributes:
            size = radius of point on the screen in pixels 
    """
    
    def __init__(self, coords, **kwargs):
        """ Creates a new point with the coordinates 'coords' of the form
            [x, y, z] specified in mm. """
        super().__init__(pos=np.array(coords), **kwargs)
    
    def draw_self(self, img, resolution, origin): ### TODO Implement opacity? 3D? Coordinate systems?
        """ Draws this point on the image specified.
        
        See Entity.draw_self
        """
        
        cv2.circle(img, (int(origin[0] + self.gpos()[0] * resolution), 
                         int(origin[1] - self.gpos()[1] * resolution)), 
                         self.attributes["size"], self.attributes["color"], 
                         thickness=-1,
                         lineType=cv2.LINE_AA)

class P(Point):
    """ Alias for Point. """
    pass
    
           
class Line(Entity):
    """ A line of infinite length. 
    
    Attributes:
        p: Point; a point on the line
        slope: 1d 3-element Numpy array; A unit vector parallel to line 
        size = thickness of line on the screen in pixels
    """
    
    def __init__(self, coords, slope, **kwargs):
        """ Creates a new Line. 
        
        Args:
            coords: [x, y, z] list; coordinates of a point on the line
            m: [x, y, z] list; A unit vector in the direction of the line
        """
        super().__init__(**kwargs)
        self.attributes["p"] = Point(coords)
        self.attributes["slope"] = np.array(slope)
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this line on the image specified.
        
        See Entity.draw_self
        """
        stretch_factor = 999999 # to create stretched line endpoints that 
                                # will exceed the image window (sloppy)
        p0_stretched = self.attributes["p"].gpos() + \
                stretch_factor * self.attributes["slope"]
        p1_stretched = self.attributes["p"].gpos() - \
                stretch_factor * self.attributes["slope"]
        cv2.line(img, (int(origin[0] + p0_stretched[0] * resolution),
                       int(origin[1] - p0_stretched[1] * resolution)),
                      (int(origin[0] + p1_stretched[0] * resolution),
                       int(origin[1] - p1_stretched[1] * resolution)),
                       self.attributes["color"], 
                       thickness=self.attributes["size"], 
                       lineType=cv2.LINE_AA)

class Line_Seg(Entity):
    """ A segment of line between two points. 
    
    Attributes:
        p: Point; start point on line segment
        q: Point; end point on long segment
        slope: 1d 3-element Numpy array; A unit vector parallel to line 
        size = thickness of line on the screen in pixels
    """
    
    def __init__(self, coords_start, coords_end, **kwargs):
        """ Creates a new Line Segment. 
        
        Args:
            coords_start: [x, y, z] list; coordinates of a point on the line.
            coords_end: [x, y, z] list; coordinates of a point on the line.
        """
        super().__init__(**kwargs, components=[Point(coords_start),
                                               Point(coords_end)])
        
    def start(self):
        """ Returns the start point on this Line Segment.
        
        Returns:
            Point (Entity) at start of line segment.
        """
        return self.attributes["components"][0]
        
    def end(self):
        """ Returns the end point on this Line Segment.
        
        Returns:
            Point (Entity) at end of Line Segment.
        """
        return self.attributes["components"][1]
    
    def length(self):
        """ Returns the length of this Line Segment. 
        
        Returns:
            A float containing the distance between the two endpoints.
        """
        return ((self.end().pos()[0] - self.start().pos()[0]) ** 2 + 
                (self.end().pos()[1] - self.start().pos()[1]) ** 2 + 
                (self.end().pos()[2] - self.start().pos()[2]) ** 2) ** 0.5
    
    def slope(self):
        """ Returns the unit vector in direction from start point to end point.
        Unit vector is in local parent coordinate system. 
        
        Returns:
            Shape (3) numpy array, representing unit vector from start to end.
        """
        return np.array(self.end().pos() - self.start().pos()) / self.length()
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this line on the image specified.
        
        See Entity.draw_self
        """
        cv2.line(img, (int(origin[0] + self.start().gpos()[0] * resolution),
                       int(origin[1] - self.start().gpos()[1] * resolution)),
                      (int(origin[0] + self.end().gpos()[0] * resolution),
                       int(origin[1] - self.end().gpos()[1] * resolution)),
                       self.attributes["color"], 
                       thickness=self.attributes["size"], 
                       lineType=cv2.LINE_AA)


class Edge(Entity):
    """ A 1D curve. """
    pass
        

class Face(Entity):
    """ A 2D surface. """
    pass
    

class Circle(Face):
    """ A 2D planar circular region. """
    
    def __init__(self, radius, **kwargs):
        """ Creates a new Circle in the xy-plane.
        
        Args:
            radius: radius of circle in mm. """
        super().__init__(**kwargs)
        self.attributes["radius"] = radius
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this Circle on the image specified.
        
        See Entity.draw_self
        """
        cv2.circle(img, (int(origin[0] + self.gpos()[0] * resolution), 
                         int(origin[1] - self.gpos()[1] * resolution)), 
                         int(self.attributes["radius"] * resolution), 
                         self.attributes["color"], 
                         thickness=-1,
                         lineType=cv2.LINE_AA)
        
    
class Polygon(Face):
    """ A 2D planar polygonal surface. """
    
    def __init__(self, points, **kwargs):
        """ Creates a new polygon from the list 'points' given. 
        
        Args:
            points: (list) vertices of the polygon. 
        """
        super().__init__(components=points, **kwargs)
            
    def get_points(self):
        """ Returns all points of this polygon, in global coordinate system.

        returns:
            Shape (1, 1, 2) nmpy array, [x, y] coords on axis 2
        """
        points_list = []
        for point in self.attributes["components"]:
            points_list.append([point.gpos()[0], point.gpos()[1]]) # TODO 3D
        return np.array([points_list])
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this Polygon on the image specified.
        
        See Entity.draw_self
        """
        pixel_points = (self.get_points() * [1, -1] * resolution + origin) \
                               .astype("int32")
                # Need to flip sign on y values for proper plotting
        cv2.fillPoly(img, pixel_points, self.attributes["color"],
                lineType=cv2.LINE_AA)


### Custom-designed functional Entities. """

class Dial_Indicator(Entity):
    """ A manual a dial indicator with gauge and needle. 
    
    Attributes:
        deflection: 
            A float from 0 to 100, representing the needle deflection.
            Needle is 12 o'clock at value 0, and increases clockwise.
        diameter:
            A float in mm representing the diameter of dial indicator that the
            needle will fit inside.
        needle:
            Entity representing the pointed indicator
        dial:
            Entity representing the static dial and gauge
    """
    ARROW_LG_SCALE = .414 # Ratio of needle length to dial diameter
    ARROW_WD_SCALE = .020 # Ratio of arrow width (at center cap) to dial dia
    CENTER_CAP_SCALE = .070 # Ratio of center cap diameter to dial diameter
    
    RIM_THICK_SCALE = .95 # Ratio of face diameter to full dial diameter
    MAJ_TICK_SCALE = .80 # Ratio of dia where major ticks begin to dial dia 
    MIN_TICK_SCALE = .85 # Ratio of dia where major ticks begin to dial dia 
    
    
    NEEDLE_COLOR = [0, 0, 255] # Red needle
    FACE_COLOR = [200, 200, 200] # Light gray gauge face
    RIM_COLOR = [50, 50, 50] # Dark gray rim
    
    def __init__(self, diameter, deflection=0, **kwargs):
        """ Creates a new dial indicator. 
        
        Args:
            diameter: (float) diameter of dial indicator to fit inside (mm) 
            deflection: (float) Needle deflection from 0 to 100 (increasing CW)
        """
        super().__init__(**kwargs)
        self.attributes["diameter"] = diameter
        
        # Needle #
        center_cap = Circle(diameter * self.CENTER_CAP_SCALE / 2.0, 
                color=self.NEEDLE_COLOR)
        arrow = Polygon([P([0, diameter * self.ARROW_LG_SCALE, 0]), 
                         P([diameter * self.ARROW_WD_SCALE / 2.0, 0, 0]), 
                         P([-diameter * self.ARROW_WD_SCALE / 2.0, 0, 0])],
                         color=self.NEEDLE_COLOR)
        self.needle = Entity(components=[center_cap, arrow], 
                color=self.NEEDLE_COLOR)
        # Dial #
        rim = Circle(diameter / 2.0, size=5, color=self.RIM_COLOR)
        face = Circle(diameter / 2.0 * self.RIM_THICK_SCALE, size=5, 
                          color=self.FACE_COLOR)
        self.dial = Entity(components=[rim, face])
        
        # Major ticks # 
        indices = 100
        for i in range(indices):
            if i % 10 == 0: # major index
                tick = Line_Seg([0, diameter * self.MAJ_TICK_SCALE / 2.0, 0], 
                            [0, diameter * (1 + self.RIM_THICK_SCALE) / 4.0, 0],
                            size=1, color=self.RIM_COLOR)
            else:
                tick = Line_Seg([0, diameter * self.MIN_TICK_SCALE / 2.0, 0], 
                            [0, diameter * (1 + self.RIM_THICK_SCALE) / 4.0, 0],
                            size=1, color=self.RIM_COLOR)
            tick.move(dori=Quaternion(axis=[0, 0, 1], 
                                      radians = i / indices * 2 * np.pi))
            self.dial.add_components(tick)
        
        # Add components
        self.add_components(self.dial, self.needle)
        
        # Set deflection
        self.set_deflection(deflection)
        
                         
    def set_deflection(self, deflection):
        """ Changes the position on the dial. 
        
        Args:
            deflection: Float from 0 - 100. Dial position, with 0 at 12 o'clock
                        and 25 at 3 o'clock.
        """
        self.attributes["deflection"] = deflection
        angle = -deflection * 2. * np.pi / 100.
        self.needle.move(ori=np.array([[np.cos(angle), -np.sin(angle), 0], 
                                       [np.sin(angle), np.cos(angle), 0], 
                                       [0, 0, 1]]))
        
        




























