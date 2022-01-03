#
# entities.dimensions.py
#
# Dimensions and leader line Entity objects. 
#


import numpy as np
from pyquaternion import Quaternion
import cv2
from entities.entity import Entity


#TODO: 3d functionality for rotated leaders. Consider using primatives instead.
class Leader(Entity):
    """ A generic leader line, with optional jogs and arrows.
    
    Attributes:
        vertices (shape(N, 2) np.array): xy segment vertices Leader CSYS. 
        start_arrow (bool): Whether arrow is present on first vertex.
        end_arrow (bool): Whether arrow is present on last vertex.
        extension (float): Fraction to be drawn, from 0-1.
    """
    
    def __init__(self, vertices, start_arrow=False, end_arrow=False, 
            extension=0., **kwargs):
        """ Creates a new Line Segment. 
        
        Args:
            vertices [[x, y]]: List of xy line segment vertices in Leader CSYS.
            start_arrow (bool): Whether arrow is present on first vertex.
            end_arrow (bool): Whether arrow is present on last vertex.
        """
        super().__init__(**kwargs)
        self.ARROW_TAPER = .25  # Half angle of arrow tip, radians. 
        self.ARROW_LENGTH = .09  # mm  #TODO: zoom independent arrow sizes?
        self.put("arrow_lg", (self.get("size") ** 0.5) * self.ARROW_LENGTH)
        self.put("vertices", np.array(vertices))
        self.put("start_arrow", start_arrow)
        self.put("end_arrow", end_arrow)
        self.put("extension", extension)
    
    def add_to_image(self, img, resolution, origin):
        """ Adds this entity on the image specified, based on its visibility.
        Don't add the leader if it has zero extension. 
        See Entity.add_to_image().
        """
        if self.get("extension") > 0:
            super().add_to_image(img, resolution, origin)
    
    def length(self):
        """ Returns the total length of all leader segments (float). """
        total_lg = 0.0
        for i in range(1, len(self.get("vertices"))):
            total_lg += (
                (self.get("vertices")[i][0] - self.get("vertices")[i-1][0]) ** 2
              + (self.get("vertices")[i][1] - self.get("vertices")[i-1][1]) ** 2
            ) ** 0.5
        return total_lg
    
    def get_frac_vertices(self):
        """ Returns vertices of line segments to be drawn, given extension.
        
        Returns:
           Shape (N, 3) np.array with [x, y, z] coords of segment vertices.
        """
        if len(self.get("vertices")) == 0 or self.get("extension") < 1E-8:
            return np.zeros((0, 3))

        verts = np.copy(self.get("vertices"))
        output = np.copy([verts[0]])
        
        if len(self.get("vertices")) == 1:
            return output
            
        # Aesthetic arrow compensation: Add verticies at the triangle bases:
        if self.get("start_arrow"):
            new_start = verts[0] + (self.get("arrow_lg") * 
                (verts[1] - verts[0]) / np.linalg.norm(verts[1] - verts[0]))
            verts = np.insert(verts, 1, new_start, axis=0)
        if self.get("end_arrow"):
            new_end = verts[-1] + (self.get("arrow_lg") * 
                (verts[-2] - verts[-1]) / np.linalg.norm(verts[-2] - verts[-1]))
            verts[-1] = new_end
        
        total_lg = self.length()
        current_lg = 0.0
        
        for i in range(1, len(verts)):
            curr = verts[i]
            prev = verts[i-1]
            
            delta_lg = np.linalg.norm(curr - prev)
            
            if current_lg + delta_lg <= total_lg * self.get("extension"):
                output = np.vstack((output, curr))
                current_lg += delta_lg
            else:
                # Go partway of distance between prev and curr
                dir = (curr - prev) / np.linalg.norm(curr - prev)
                partway_lg = self.get("extension") * total_lg - current_lg
                partial = prev + dir * partway_lg
                output = np.vstack((output, partial))
                break
        
        if self.get("start_arrow"):
            output = output[1:]
            
        return output
        
    def get_start_arrow(self):
        """ Computes the vertices of the triangular start arrow.
        Shrinks the triangle if extension is small enough and has not passed it. 
        
        Returns:
            Shape(1, 3, 3) np.array: [[[xyz1], [xyz2], [xyz3]]]
        """ 
        if len(self.get("vertices")) < 2:
            return np.array(np.zeros((1, 0, 3)))
        
        bounded_lg = min(self.get("extension") * self.length(), 
                       self.get("arrow_lg"))
        arrow_half_wd = bounded_lg * np.tan(self.ARROW_TAPER)
        
        start = self.get("vertices")[0]
        next = self.get("vertices")[1]
        e = (next - start) / np.linalg.norm(next - start)
        n = np.array([-e[1], e[0], e[2]])  #TODO 3D
        
        vert1 = start + bounded_lg * e + arrow_half_wd * n
        vert2 = start + bounded_lg * e - arrow_half_wd * n
        
        return np.array([[start, vert1, vert2]])
    
    def get_end_arrow(self):
        """ Computes the vertices of the triangular end arrow.
        Truncates to a trapezoid if extension is small and has not passed it. 
        
        Returns:
            Shape(1, 4, 3) np.array: [[[xyz1], [xyz2], [xyz3], [xyz4]]]
        """ 
        if len(self.get("vertices")) < 2:
            return np.array(np.zeros((1, 0, 3)))
        
        trap_ht = (self.get("extension") * self.length() - 
                (self.length() - self.get("arrow_lg")))
        if trap_ht <= 0:
            return np.array(np.zeros((1, 0, 3)))
            
        base_width = self.get("arrow_lg") * np.tan(self.ARROW_TAPER)
        mid_width = (self.get("arrow_lg") - trap_ht) * np.tan(self.ARROW_TAPER)
        
        end = self.get("vertices")[-1]
        prev = self.get("vertices")[-2]
        e = (end - prev) / np.linalg.norm(end - prev)
        n = np.array([-e[1], e[0], e[2]])  #TODO 3D
        
        base1 = (self.get("vertices")[-1] - self.get("arrow_lg") * e
                + base_width * n)
        base2 = (self.get("vertices")[-1] - self.get("arrow_lg") * e
                - base_width * n)
        mid1  = (self.get("vertices")[-1] - (self.get("arrow_lg") - trap_ht) * e
                + mid_width * n)
        mid2  = (self.get("vertices")[-1] - (self.get("arrow_lg") - trap_ht) * e
                - mid_width * n)
        
        return np.array([[base1, base2, mid2, mid1]])
    
    def draw_self(self, img, resolution, origin): 
        """ Draws this line on the image specified.
        
        See Entity.draw_self
        """
        frac_vertices = self.get_frac_vertices()
        
        # Draw lines:
        for i in range(1, len(frac_vertices)):
            start = frac_vertices[i-1]
            end = frac_vertices[i]

            cv2.line(img, (int(origin[0] + start[0] * resolution),
                           int(origin[1] - start[1] * resolution)),
                          (int(origin[0] + end[0] * resolution),
                           int(origin[1] - end[1] * resolution)),
                           self.get("color"), 
                           thickness=self.get("size"), 
                           lineType=cv2.LINE_AA)

        # Draw arrows:
        if self.get("start_arrow"):
            pixel_points = (self.get_start_arrow()[:,:,:2] * [1, -1] * 
                    resolution + origin).astype("int32")
            cv2.fillConvexPoly(img, pixel_points, self.attributes["color"],
                    lineType=cv2.LINE_AA)
                    
        if self.get("end_arrow"):
            pixel_points = (self.get_end_arrow()[:,:,:2] * [1, -1] * 
                    resolution + origin).astype("int32")
            cv2.fillConvexPoly(img, pixel_points, self.attributes["color"],
                    lineType=cv2.LINE_AA)  
    
    
    
    
    
    
    
