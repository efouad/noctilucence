#
# Helper functions.
#

import numpy as np
import matplotlib.pyplot as plt
from sympy import Point as SymPoint
from sympy import Polygon as SymPolygon
from entities.primatives import *




def span(start, end, n_elems, profile="sigmoid"):
    """ Returns a np.array spanning a range of elements between a start and end.
    
    Elements of output array are derived from evenly spaced x values
    modified by the profile function specified (see below). 
    
    Args:
        start:    (float or np.array)  Starting value in span array
        end:      (float or np.array)  Ending value in span array
        n_elems:                (int)  Number of elements in span array
        profile:             (string)  Type of span array distribution.
                                       Options: "linear", "quadratic", 
                                       "neg_quadratic", "sinusoid", "sigmoid"
    
    Returns:
        A shape (n_elems, np.shape(start)) Numpy array, with elements spanning 
        the specified range and profile. 
    """
    
    # For one element, we always want to reflect the final value.
    if n_elems <= 1:
        shape = np.array([1.])
    else:    
        # Create distribution shape, from 0 - 1:
        if profile == "linear":
            shape = np.linspace(0, 1, n_elems)
        elif profile == "quadratic":
            shape = (np.linspace(0, 1, n_elems)) ** 2
        elif profile == "neg_quadratic":
            shape = 1 - (np.linspace(1, 0, n_elems)) ** 2
        elif profile == "sinusoid":
            shape = 0.5 - np.cos(np.linspace(0, np.pi, n_elems)) / 2
        else: #if profile == "sigmoid":
            shape = 1/(1 + np.exp(-10 * np.linspace(-.5, .5, n_elems)))
        # Artificially round end points:
        shape -= shape[0]
        shape *= 1.0 / shape[-1]
    
    return start + np.outer(shape, (end - start))




def convex(poly):
    """ Returns whether polygon points form a convex shape. 
    
    Returns:
        (boolean) convexity status of Polygon points. 
    """
    sympoints = []
    for point in poly.attributes["components"]:
        sympoints.append(SymPoint((point.pos()[0], point.pos()[1])))
    sympoly = SymPolygon(*sympoints)
    return sympoly.is_convex()




def dist_to_poly(poly, point, dir):
    """ Gives distance from point to polygon surface, in direction specified.
    #TODO - currently only implemented for 2d (x, y, 0)
    
    Args:
        poly    Polygon              Polygon Entity to measure distance to
        point   Point                Reference point to measure distance from
        dir     Shape (3) np.array   Direction vector from point, normalized
    
    Returns:
        float:  Positive fwd distance from point -> poly in specified direction.
                np.inf if no forward intersection.
                Negative fwd dist if both forward and a backwards intersection
                -np.inf if backwards intersection, but no forwards intersection
    """
    
    # Ensure dir is normalized
    a = dir / np.linalg.norm(dir)
    
    # Construct normal to direction line
    n = np.array([-a[1], a[0], 0])
    
    # Measure distance of each line segment to the ray coming from point.  
    # Ignore line segments that do not intersect point. 
    min_line_fwd_dist = np.inf
    inside_factor = 1
    vertices = poly.get_points()[0]
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i+1) % len(vertices)]
        norm_start = np.dot(n, start - point.gpos())
        norm_end = np.dot(n, end - point.gpos())
        ax_start = np.dot(a, start - point.gpos())
        ax_end = np.dot(a, end - point.gpos())
        
        if norm_start * norm_end > 0:
            continue  # same sign; no intersection
        else:
            fwd_dist = ax_start + ((0 - norm_start) / (norm_end - norm_start) * 
                    (ax_end - ax_start))
            if fwd_dist < 0:
                inside_factor = -1  # must be inside polygon
                continue
            if fwd_dist < min_line_fwd_dist:
                min_line_fwd_dist = fwd_dist
    
    return inside_factor * min_line_fwd_dist
    
    
if __name__ == '__main__':
    
    # Dist to poly testing
    poly = Polygon([P([-1, -1, 0]), 
                    P([-1, -4, 0]), 
                    P([ 4, -4, 0]), 
                    P([ 4,  1, 0]),
                    P([ 2, -2, 0])])
    point = P([2, -.25, 0])
    dir = [.5, -1, 0]
    print()
    print(dist_to_poly(poly, point, dir))
    
    # Profile shape plotting
    ##shape = 1/(1 + np.exp(-15 * np.linspace(-.5, .5, 100)))
    ##plt.plot(shape)
    ##plt.xlabel("x")
    ##plt.ylabel("y")
    ##print(shape)
    ##plt.show()