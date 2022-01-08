#
# 2d_flatness_visualizer.py
#
# Solves for minimum zone flatness of a set of 2d points.
# Visualizes and animates the candidate solutions. 
#



# Add top level package to path
import sys
import os
from os import path
local_path = os.path.abspath(__file__)
pack_name = "noctilucence"
pack_path = local_path[:local_path.index(pack_name) + len(pack_name)]
sys.path.insert(0, pack_path)

import cv2
import numpy as np

import animator
from animator import animation as ani
from animator.scene import Scene
from animator import funcs

from animator.entities.primatives import Point, Line



def load_data(filename):
    """ Imports a string 'filename' of a csv file of 2d points, and returns
        the data in a np array. """
    return np.genfromtxt(filename, delimiter=',', dtype=np.float64)

def calc_flatness(points, data=False):
    """ Returns minimum zone flatness of np array 'points' specified.
        Iterates with parallel lines across every candidate position in 
        the convex hull of the points & selects the one with min separation. 
        If 'data' = True, returns a tuple of the form (flatness, 
        flatness_step_vals, slope_vals, p0_vals, p1_vals), where 
        flatness_step_vals, slope_vals, p0_vals, and p1_vals are numpy arrays 
        containing respectively the flatness, flatness line slope, lower point, 
        and upper point of each evaluated roll position in the calculation. """
    
    # Extract two points with extreme y values; these will be used as initial
    # starting points in the flatness search. 
    n0 = np.argmin(points, axis=0)[1] # index of min y value
    n1 = np.argmax(points, axis=0)[1] # index of max y value
    p0 = points[n0] # Point with minimum y value
    p1 = points[n1] # Point with maximum y value
    
    # Horizontal line -- original flatness line estimate
    p0_slope = np.array([1, 0]) # unit vector with slope of flatness line
    p1_slope = -p0_slope
    
    # Results to be tabulated from each roll position
    flatness_step_vals = np.array([])
    slope_vals = np.empty((0, 2))
    p0_vals = np.empty((0, 2))
    p1_vals = np.empty((0, 2))
    
    # Loop over all candidate flatness values by rolling around point set, 
    # similar to string wrapping algorithm to find a convex hull.
    # Stop when rolled around completely such that p0 equals the original p1 
    # point or vice versa.
    while (not np.array_equal(p0, points[n1])) and \
          (not np.array_equal(p1, points[n0])): # Stop when rolled to other side
        # Calculate normalized direction from current point to each other point.
        p0_dirs = np.empty((0, 2)) # list of 2d unit vectors
        p0_dot_prods = np.array([]) # list of normalized dot products, -1 to 1
        p1_dirs = np.empty((0, 2)) # list of 2d unit vectors
        p1_dot_prods = np.array([]) # list of normalized dot products, -1 to 1

        # Calculate directions of p0 and p1 to each other point, and dot
        # products of these directions with their current slopes.
        for i in range(np.size(points, 0)):
            point = points[i]
            # Assess p0
            if not np.array_equal(point, p0): # Don't assess existing p0
                p0_dist = ((point[0] - p0[0]) ** 2 + (point[1] - p0[1]) ** 2) \
                            ** .5
                p0_dir = np.array([(point[0] - p0[0]) / p0_dist, 
                                   (point[1] - p0[1]) / p0_dist])
                p0_dirs = np.concatenate((p0_dirs, np.array([p0_dir])), axis=0)
            else: # if point = p0
                p0_dirs = np.concatenate((p0_dirs, np.array([[0, 0]])), axis=0)
            p0_dot_prods = np.append(p0_dot_prods, np.dot(p0_slope, p0_dir))
            
            # Assess p1
            if not np.array_equal(point, p1): # Don't assess existing p1
                p1_dist = ((point[0] - p1[0]) ** 2 + (point[1] - p1[1]) ** 2) \
                            ** .5
                p1_dir = np.array([(point[0] - p1[0]) / p1_dist, 
                                   (point[1] - p1[1]) / p1_dist])
                p1_dirs = np.concatenate((p1_dirs, np.array([p1_dir])), axis=0)
            else: # if point = p1
                p1_dirs = np.concatenate((p1_dirs, np.array([[0, 0]])), axis=0)
            p1_dot_prods = np.append(p1_dot_prods, np.dot(p1_slope, p1_dir))
        
        # Find which dot product is the largest, and update p0 or p1 to the new
        # point (rolls onto it).
        # Update slopes of the flatness lines.  
        p0_max_dp = np.max(p0_dot_prods)
        p0_max_dp_index = np.argmax(p0_dot_prods)
        p1_max_dp = np.max(p1_dot_prods)
        p1_max_dp_index = np.argmax(p1_dot_prods)
        
        if p0_max_dp >= p1_max_dp:
            p0 = points[p0_max_dp_index]
            p0_slope = p0_dirs[p0_max_dp_index]
            p1_slope = -p0_slope
        else:
            p1 = points[p1_max_dp_index]
            p1_slope = p1_dirs[p1_max_dp_index]
            p0_slope = -p1_slope
        
        # Report flatness from this step.
        p01 = (p1 - p0) # vector from p0 to p1
        flat_normal = np.array([-p0_slope[1], p0_slope[0]]) # rotate slope 90deg
        flatness_step_val = abs(np.dot(p01, flat_normal))
        
        # Tabulate data
        flatness_step_vals = np.append(flatness_step_vals, flatness_step_val)
        slope_vals = np.concatenate((slope_vals, np.array([np.copy(p0_slope)])), 
                axis=0)
        p0_vals = np.concatenate((p0_vals, np.array([np.copy(p0)])), axis=0)
        p1_vals = np.concatenate((p1_vals, np.array([np.copy(p1)])), axis=0)
        
    # Report flatness
    flatness = np.min(flatness_step_vals)
    if data:
        return (flatness, flatness_step_vals, slope_vals, p0_vals, p1_vals)
    else:
        return flatness

def plot_point(img, point, resolution, origin, color=[255, 255, 255]):
    """ Plots the points specified on an image. 
    
    Args:
        img: 3d Numpy array; image upon which points will be plotted
        points: 2-element; coordinates of point to plot
        resolution: px/mm; pixels on image per millimeter of entity size 
        origin: list [x_origin, y_origin] of image origin in pixels
        color: list [blue, green, red], uint8); color of the plotted point
    """
    plotted_point = Point([point[0], point[1], 0], size=4, color=color)
    plotted_point.draw_self(img, resolution, origin)
    
def draw_line(img, point, slope, resolution, origin, color=[255, 255, 255]):
    """ Plots the line specified on an image. 
    
    Args:
        img: 3d Numpy array; image upon which points will be plotted
        point: 2-element Numpy array; a point on the line
        slope: 2-element Numpy array; a unit vector in the line direction
        resolution: px/mm; pixels on image per millimeter of entity size 
        origin: list [x_origin, y_origin] of image origin in pixels
        color: list [blue, green, red], uint8); color of the plotted line
    """
    plotted_line = Line([point[0], point[1], 0], [slope[0], slope[1], 0], 
                        size=1, color=color)
    plotted_line.draw_self(img, resolution, origin)
    

def draw_image(points, line_points_0, line_points_1, slopes):
    """ Creates an image with the points and lines given. 
    
    Args:
        points: 2d Numpy array; list of points to plot
        line_points_0: 2d Numpy array; a point on the lower plotted lines
        line_points_1: 2d Numpy array; a point on the upper plotted lines
        slopes: 2d Numpy array; unit vectors on each of the plotted lines
    """
    
    img = np.zeros([1000, 1500, 3], np.uint8)  # blank bgr image
    resolution = 5  # px / mm
    origin = [500, 1000]
    
    for point in points:
        plot_point(img, point, resolution, origin)
    
    img_points_only = np.copy(img)
    
    for i in range(np.shape(line_points_0)[0]):
        img_points_and_line = np.copy(img_points_only)
        draw_line(img_points_and_line, line_points_0[i], slopes[i], resolution, 
                origin, color=[0, 255, 0])
        draw_line(img_points_and_line, line_points_1[i], slopes[i], resolution, 
                origin, color=[0, 0, 255])
        cv2.imshow('plot', img_points_and_line)
        cv2.waitKey(150)
    # Cycle image: 
    draw_image(points, np.flip(line_points_0, axis=0), 
                       np.flip(line_points_1, axis=0),
                       np.flip(slopes, axis=0))
    

if __name__ == '__main__':
    points = load_data('2d_points_in.csv')
    flatness, flatness_step_vals, slope_vals, p0_vals, p1_vals = calc_flatness(
            points, data=True)
    print("FLATNESS: " + str(flatness) + "\n")
    print("Flatness steps: \n" + str(flatness_step_vals))
    print("Slope values: \n" + str(slope_vals))
    print("p0_vals: \n" + str(p0_vals))
    print("p1_vals: \n" + str(p1_vals))
    print()
    draw_image(points, p0_vals, p1_vals, slope_vals)

