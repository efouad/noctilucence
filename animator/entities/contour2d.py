#
# entities.contour2d.py
#
# A 2D contour, representing the periphery of a solid or hole.
#

from .entity import Entity
##from entity import Entity
import numpy as np
import cv2

# TODO 3D
class Contour2D(Entity):
    """ A 2d contour of lines and arcs. 
        
    Attributes:
        size (int): thickness of contour
    """
    
    def __init__(self, contours=[], jaggedness=0.0, n_points=30, seed=0, 
            **kwargs):
        """ Creates a new contour, consisting of a list of child contours. 
        
        Args:
            contours (list of Contour2D): contours contained within this one
            jaggedness (float): Max distance to protrude from nominal contour
            n_points (int): Number of points to divide each sub-contour into
            seed (int): Random seed to control jaggedness
        """
        super().__init__(components=contours, **kwargs)
        self.put("jaggedness", jaggedness)
        self.put("n_points", n_points)
        self.put("seed", seed)
                # seed ensures jaggedness always stays the same during render
    
    def draw_self(self, img, resolution, origin): #
        """ Draws this contour on the image specified.
        
        See Entity.draw_self
        """
        # # for contour in self.get("components"):
            # # contour.draw_self(img, resolution, origin)
            
        points = self.jagged_points()
        points = np.array([origin]) + points * [1, -1] * resolution  # px
        # cv2.polylines(img,
                      # np.int32([points]),
                      # isClosed=False,
                      # color=self.get("color"), 
                      # thickness=self.get("size"), 
                      # lineType=cv2.LINE_AA)
        cv2.drawContours(img,
                         np.int32([points]),
                         0,
                         color=self.get("color"),
                         thickness=self.get("size"),
                         # thickness=cv2.FILLED,
                         lineType=cv2.LINE_AA)
                         
    def segments(self):
        """ Returns points and normals of a segmentation into n points.
        
        Returns: 
            A 2-tuple containing
                points (np.array((n_points, 2))): Points of segmented line
                normals (np.array((n_points, 2))): Normal vectors at each point
        """
        points = np.zeros((0, 2))
        normals = np.zeros((0, 2))
        for contour in self.get("components"):
            p, n = contour.segments()
            points = np.vstack((points, p))
            normals = np.vstack((normals, n))
        return points, normals
        
    def jagged_points(self):
        """ Returns the points along the contour if it were made jagged.
        
        Returns:
            points (np.array(n_points, 2)): Array of jagged points
        """
        np.random.seed(self.get("seed"))
        points, normals = self.segments()
        # Set random offsets: (initial and final points have 0 offset)
        offsets = np.random.uniform(-self.get("jaggedness"), 
                self.get("jaggedness"), (len(points), 1))
        offsets[0] = [0]
        offsets[-1] = [0]
        # Offset points in direction of normals:
        jagged_points = points + offsets * normals
        return jagged_points



class LineContour2D(Contour2D):
    """ A 2D line contour. """
    
    def __init__(self, start, end, **kwargs):
        """ Creates a new line contour.
        
        Args:
            start: [x, y] start of line segment, mm
            end: [x, y] end of line segment, mm
        """
        super().__init__([self], **kwargs)
        self.put("start", np.array(start))
        self.put("end", np.array(end))
        
    def draw_self(self, img, resolution, origin): 
        """ Draws this contour on the image specified. 
        
        See Entity.draw_self
        """
        if self.get("jaggedness") > 0:
            super().draw_self(img, resolution, origin)
        else:
            points = np.array([self.get("start"), self.get("end")])
            points = np.array([origin]) + points * [1, -1] * resolution  # px
            cv2.polylines(img,
                          np.int32([points]),
                          isClosed=False,
                          color=self.attributes["color"], 
                          thickness=self.attributes["size"], 
                          lineType=cv2.LINE_AA)
    
    def segments(self):
        """ Returns points and normals of a segmentation into n points.
        
        See Contour2D.segments
        """
        np.random.seed(self.get("seed"))
        spaces = np.random.dirichlet(10 * np.ones(self.get("n_points") - 1))
        spaces = np.hstack([0, spaces])
                # spaces has n_points entries that sum to 1
        points = (np.tile(self.get("start"), (len(spaces), 1)) +
                        np.multiply.outer(np.cumsum(spaces), 
                                (self.get("end") - self.get("start"))))

        normal = (np.array([[0, -1], [1, 0]]) @ (self.get("end") - 
                self.get("start")) / np.linalg.norm(self.get("end") - 
                self.get("start")))
        normals = np.linspace(normal, normal, self.get("n_points"))
        return (points, normals)
    

class ArcContour2D(Contour2D):
    """ A 2D arc contour. """
    
    def __init__(self, center, radius, start_ang, end_ang, **kwargs):
        """ Creates a new arc contour.
        
        Args:
            center: [x, y] (float): center of arc, mm
            radius: (float): radius of arc, mm
            start_ang: (float) start angle of arc, rad
            end_ang: (float) end angle of arc, rad
        """
        super().__init__([self], **kwargs)
        self.put("ctr", np.array(center))
        self.put("rad", radius)
        self.put("ang1", start_ang)
        self.put("ang2", end_ang)
        

    def draw_self(self, img, resolution, origin): 
        """ Draws this contour on the image specified. Optionally show jagged.
        
        See Entity.draw_self
        """
        if self.get("jaggedness") > 0:
            super().draw_self(img, resolution, origin)
        else:
            cv2.ellipse(img, 
                        (int(origin[0] + self.get("ctr")[0] * resolution), 
                         int(origin[1] - self.get("ctr")[1] * resolution)),
                        (int(self.get("rad") * resolution), 
                         int(self.get("rad") * resolution)),
                        0, 
                        -np.rad2deg(self.get("ang1")),
                        -np.rad2deg(self.get("ang2")),
                        color=self.get("color"), 
                        thickness=self.get("size"), 
                        lineType=cv2.LINE_AA)
    
    def segments(self):
        """ Returns points and normals of a segmentation into n points.
        
        See Contour2D.segments
        """
        np.random.seed(self.get("seed"))
        spaces = np.random.dirichlet(10 * np.ones(self.get("n_points") - 1))
        spaces = np.hstack([0, spaces])
                # spaces has n_points entries that sum to 1
        angs = np.array([self.get("ang1") + 
                (self.get("ang2") - self.get("ang1")) * np.cumsum(spaces)]).T
        points = (self.get("ctr") + 
                self.get("rad") * np.hstack([np.cos(angs), np.sin(angs)]))
        normals = np.hstack([np.cos(angs), np.sin(angs)])
        return (points, normals)
                      

class CirContour2D(ArcContour2D):
    """ A 2D circle contour. """
    
    def __init__(self, center, radius, **kwargs):
        """ Creates a new circle contour.
        
        Args:
            center: [x, y] (float): center of arc, mm
            radius: (float): radius of arc, mm
        """
        super().__init__(center, radius, np.pi/4, 9*np.pi/4, **kwargs)


if __name__ == "__main__":
    img = np.zeros([500, 500, 3])
    
    # Test renderings of multi-edge contour
    contours = []
    contours.append(ArcContour2D([100., -100.], 50., 5*np.pi/4, np.pi/2, 
            jaggedness=3, n_points=60))
    contours.append(LineContour2D([100., -50.], [300., -50],
            jaggedness=3, n_points=60))
    contours.append(ArcContour2D([300., -100.], 50., np.pi/2, 0, 
            jaggedness=5, n_points=40))
    contours.append(LineContour2D([350., -100.], [350., -300],
            jaggedness=5, n_points=80))
    contours.append(ArcContour2D([300., -300.], 50., 0, -3*np.pi/4, 
            jaggedness=5, n_points=40))
    contours.append(LineContour2D(
            [300.-np.sqrt(2)/2*50, -300.-np.sqrt(2)/2*50], 
            [100.-np.sqrt(2)/2*50, -100-np.sqrt(2)/2*50],
            jaggedness=0, n_points=50))
    contour = Contour2D(contours=contours, jaggedness=4.0, n_points=2, seed=0)
    contour.draw_self(img, 1, [0, 0])
    
    # Test rendering of circle
    circ = CirContour2D([50, -200], 40, jaggedness=3, n_points=200)
    circ.draw_self(img, 1, [0, 0])

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    # # Test renderings of jagged lines and arcs
    # lin = LineContour2D([100., -137.], [300., -400.], jaggedness=40., 
            # n_points=60)
    # points, normals = lin.segments()
    # jagged_points = lin.jagged_points()
    
    # arcs = []
    # arcs.append(CirContour2D([350., -250.], 50., jaggedness=10, n_points=20))
    # arcs.append(CirContour2D([350., -250.], 50.))
    # arcs.append(ArcContour2D([300., -50.], 20., np.pi/4, 3*np.pi/4))
    # arcs.append(ArcContour2D([300., -100.], 20., np.pi/4, 2 * np.pi + np.pi/8))
    # arcs.append(ArcContour2D([125, -125], 150, .25, 4.45))
    # a = (ArcContour2D([125, -125], 150, .25, 4.45, 
            # jaggedness=10, n_points=100))  
    # arcs.append(a)
    # a_points, a_normals = a.segments()
    # print(a_points)
    # print(a_normals)
    # for i in range(len(a_points)):
        # p = a_points[i,:] - a.get("ctr")
        # print("--")
        # print(p / np.linalg.norm(p))
        # print(a_normals[i,:])
    
    # img = np.zeros([500, 500, 3])
    # lin.draw_self(img, 1, [50, 50])
    # for arc in arcs:
        # arc.draw_self(img, 1, [50, 50])
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

