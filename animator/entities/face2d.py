#
# entities.face2d.py
#
# A flat face, whose periphery is defined by (possibly multiple) contours
#


from .entity import Entity
from .contour2d import Contour2D, ArcContour2D, LineContour2D, CirContour2D
##from entity import Entity
##from contour2d import Contour2D, ArcContour2D, LineContour2D, CirContour2D
import numpy as np
import cv2

# TODO 3D
class Face2D(Entity):
    """ A 2d flat face. """
    
    def __init__(self, contours=[], **kwargs):
        """ Creates a new contour, consisting of a list of child contours. 

        Args:
            contours (list of Contour2D): Periphery contours (inside / outside)
        """
        super().__init__(components=contours, **kwargs)
    
    def draw_self(self, img, resolution, origin): #
        """ Draws this contour on the image specified.
        
        See Entity.draw_self
        """
        points_list = []
        for contour in self.get("components"):
            points = contour.jagged_points()
            points = np.array([origin]) + points * [1, -1] * resolution  # px
            ##points_list.append(points)
            points_list.append(np.int32(points))
        cv2.drawContours(img,
                         points_list,
                         ##np.int32(points_list),
                         -1,
                         color=self.get("color"),
                         thickness=cv2.FILLED,#self.get("size"),
                         lineType=cv2.LINE_AA)


if __name__ == "__main__":
    img = np.zeros([500, 500, 3])
    
    # Outer contour
    contours = []
    contours.append(ArcContour2D([100., -100.], 50., 5*np.pi/4, np.pi/2))
    contours.append(LineContour2D([100., -50.], [300., -50]))
    contours.append(ArcContour2D([300., -100.], 50., np.pi/2, 0))
    contours.append(LineContour2D([350., -100.], [350., -300]))
    contours.append(ArcContour2D([300., -300.], 50., 0, -3*np.pi/4))
    contours.append(LineContour2D(
            [300.-np.sqrt(2)/2*50, -300.-np.sqrt(2)/2*50], 
            [100.-np.sqrt(2)/2*50, -100-np.sqrt(2)/2*50]
    ))
    outer = Contour2D(contours=contours, jaggedness=5.0, seed=0)
    
    # Inner contour 1
    inner1 = CirContour2D([250., -200.], 40, jaggedness=3.0, n_points=100)
    
    # Inner contour 2
    contours = []
    contours.append(LineContour2D([200, -75], [250, -75]))
    contours.append(LineContour2D([250, -75], [250, -125]))
    contours.append(LineContour2D([250, -125], [200, -125]))
    contours.append(LineContour2D([200, -125], [200, -75]))
    inner2 = Contour2D(contours=contours, jaggedness=2.0)
    
    face = Face2D(contours=[outer, inner1, inner2])
    face.draw_self(img, 1, [0, 0])
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

