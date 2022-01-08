#
# entities.primatives.py
#
# Primative geometric Entity objects: points, lines, faces, etc.
#


import numpy as np
from pyquaternion import Quaternion
import cv2
from .entity import Entity
from ..funcs import convex


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
    
    
class Edge(Entity):
    """ A 1D curve. """
    
    def __init__(self, **kwargs):
        """ Creates a new edge.
        """
        super().__init__(**kwargs)
    
           
class Line(Edge):
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
        p0_stretched = (self.attributes["p"].gpos() +
                stretch_factor * self.attributes["slope"])
        p1_stretched = (self.attributes["p"].gpos() -
                stretch_factor * self.attributes["slope"])
        cv2.line(img, (int(origin[0] + p0_stretched[0] * resolution),
                       int(origin[1] - p0_stretched[1] * resolution)),
                      (int(origin[0] + p1_stretched[0] * resolution),
                       int(origin[1] - p1_stretched[1] * resolution)),
                       self.attributes["color"], 
                       thickness=self.attributes["size"], 
                       lineType=cv2.LINE_AA)


class Line_Seg(Edge):
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


class Circle(Edge):
    """ A 1D circular curve. """
    
    def __init__(self, radius, **kwargs):
        """ Creates a new Circle in the xy-plane.
        
        Args:
            radius: radius of circle in mm. """
        super().__init__(**kwargs)
        self.attributes["radius"] = radius
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this Disk on the image specified.
        
        See Entity.draw_self
        """
        cv2.circle(img, (int(origin[0] + self.gpos()[0] * resolution), 
                         int(origin[1] - self.gpos()[1] * resolution)), 
                         int(self.attributes["radius"] * resolution), 
                         self.attributes["color"], 
                         thickness=self.attributes["size"],
                         lineType=cv2.LINE_AA)
        

class Face(Entity):
    """ A 2D surface, surrounded by a border of edges. 
    
    Attributes:
        boundary: list of Edges making up the periphery of this face. 
        boundary_color: list [b, g, r] representing color of boundary edges. 
    """
    
    def __init__(self, boundary=[], boundary_color=[255, 255, 255], **kwargs):
        """ Creates a new abstract face.
        
        Args:
            boundary: list of Edges making up periphery
            boundary_color: list [b, g, r] representing color of boundary edges.
        """
        super().__init__(**kwargs)
        self.attributes["boundary"] = boundary
        self.attributes["boundary_color"] = boundary_color
                

class Disk(Face):
    """ A 2D filled planar circular region. """
    
    def __init__(self, radius, **kwargs):
        """ Creates a new Disk in the xy-plane.
        
        Args:
            radius: radius of Disk in mm. """
        super().__init__(**kwargs)
        self.attributes["radius"] = radius
        
    def draw_self(self, img, resolution, origin): 
        """ Draws an opaque filled version of this Disk on image specified.
        
        See Entity.draw_self
        """
        cv2.circle(img, (int(origin[0] + self.gpos()[0] * resolution), 
                         int(origin[1] - self.gpos()[1] * resolution)), 
                         int(self.attributes["radius"] * resolution), 
                         self.attributes["color"], 
                         thickness=-1,
                         lineType=cv2.LINE_AA)


#TODO: Extend to annulus?         
class Wedge(Face):
    """ A 2D filled planar circular wedge, with start & end angle. """
    
    def __init__(self, radius, start_ang, end_ang, **kwargs):
        """ Creates a new Wedge in the xy-plane.
        
        Args:
            radius: radius of Wedge in mm. 
            start_ang: start angle of wedge in radians.
            end_ang: end angle of wedge in radians. Should be > start_ang. """
        super().__init__(**kwargs)
        self.attributes["radius"] = radius
        self.attributes["start_ang"] = start_ang
        self.attributes["end_ang"] = end_ang
        
    def draw_self(self, img, resolution, origin): 
        """ Draws an opaque filled version of this Disk on image specified.
        
        See Entity.draw_self
        """
        cv2.ellipse(img, (int(origin[0] + self.gpos()[0] * resolution), 
                          int(origin[1] - self.gpos()[1] * resolution)), 
                         (int(self.attributes["radius"] * resolution),
                          int(self.attributes["radius"] * resolution)), 
                          0,
                          360. - self.attributes["start_ang"] * 180 / np.pi,
                          360. - self.attributes["end_ang"] * 180 / np.pi,
                          self.attributes["color"], 
                          thickness=-1,
                          lineType=cv2.LINE_AA)
    
    
class Polygon(Face):
    """ A 2D planar polygonal surface. 
    
    Attributes:
        "convex" - True if points form a convex shape.  #TODO 3d convexity
    """
    
    def __init__(self, points, **kwargs):
        """ Creates a new polygon from the list 'points' given. 
        
        Args:
            points: (list) Points of the polygon's vertices. Must be 
                           consecutively ordered according to the geometry. 
        """
        super().__init__(components=points, **kwargs)
        self.attributes["convex"] = convex(self) 
                #TODO Re-check if pts change
    
    def get_points(self):
        """ Returns all points of this polygon, in global coordinate system.
        # TODO only implemented for 2d (x, y, 0)
        returns:
            Shape (N, 3) numpy array, [x, y, 0] coords on axis 2
        """
        points_list = []
        for point in self.attributes["components"]:
            points_list.append([point.gpos()[0], point.gpos()[1], 0]) 
        return np.array([points_list])
        
    def draw_self(self, img, resolution, origin): 
        """ Draws an opaque filled version of this Polygon on image specified.
        
        See Entity.draw_self
        """
        pixel_points = (self.get_points()[:,:,:2] * [1, -1] * resolution +
                            origin).astype("int32")
        if self.attributes["convex"]:
            cv2.fillConvexPoly(img, pixel_points, self.attributes["color"],
                    lineType=cv2.LINE_AA)        
        else:
            cv2.fillPoly(img, pixel_points, self.attributes["color"],
                    lineType=cv2.LINE_AA)
                
                
class Text(Entity):
    """ A field of text. """
    
    def __init__(self, text, scale=1, **kwargs): 
        """ Creates a new text field. 
        
        Args:
            text: String of text to display.
            scale: Scale factor on text size.
        """
        super().__init__(**kwargs)
        self.attributes["text"] = text
        self.attributes["scale"] = scale
        
    def draw_self(self, img, resolution, origin): # TODO Text rotation
        cv2.putText(img, self.attributes["text"], 
                         (int(origin[0] + self.gpos()[0] * resolution),
                          int(origin[1] - self.gpos()[1] * resolution)),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         self.attributes["scale"] * resolution,
                         self.attributes["color"],
                         thickness=self.attributes["size"], 
                         lineType=cv2.LINE_AA)
    
    
    
