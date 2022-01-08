#
# main_dial.py
#
# A dial indicator performing a flatness inspection.
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

from animator.entities.dial_indicator import *
from animator.entities.primatives import P, Polygon
from animator.entities.dimensions import Leader

import csv

       

def create_polygon(csv_file):
    """ Reads point data from the file specified and creates a polygon.
    
    Args:
        csv_file: (string) url of csv file with point data
    
    Returns:
        (Polygon) Entity with three flat faces and 4th face per point data
    """
    width = 6 # width of polygon in mm
    extension = .01 # extra wd. on either end of polygon, sloping down to bottom
    thickness = .25 # height of polygon in mm
    
    # initialize with bottom points, on edge across from point-data edge:
    poly_points = [P([width + extension, -thickness, 0]), 
                   P([0 - extension, -thickness, 0])]
    
    # Read file and append each point to polygon. 
    with open("dial_ani_flatness_inputs.csv", 'r') as data_file:
        csv_data = csv.reader(data_file)      
        for row in csv_data:
            poly_points.append(P([float(row[0]), float(row[1]), 0.]))
    poly = Polygon(poly_points, color=[130, 180, 250], opacity=0)
    return poly



if __name__ == '__main__':
    
    # # # Setup animation parameters # # #
    width, height = 1920//1, 1080//1
    resolution = 272.//1 # px / mm
    fps = 60
    
    
    # # # Initialize objects # # #
    ANO_COLOR = [230, 156, 0]
    WHITE = [255, 255, 255]
    
    # Create dial # 
    dial = Dial_Indicator(1.25, deflection=0, highlight_show=False,
            readout_scale=1, opacity=0)
    dial.move(pos=[0, 1.103, 0])
    
    # Create polygon
    poly = create_polygon("dial_ani_flatness_inputs.csv")
    poly.move(pos=[-3.0, -1.8, 0])
    
    # Create title text
    title_text_1 = Text("FLATNESS MEASUREMENT", color=WHITE, 
                      scale=0.010, size=3, pos=[-1.97, 0.8, 0.0], opacity=0)
    title_text_2 = Text("WITH DIAL INDICATOR", color=WHITE, 
                      scale=0.010, size=3, pos=[-1.61, 0.4, 0.0], opacity=0)
                      
    # Create dial annotation entities
    flat_lo_leader = Leader([[0., .2130, 0.], 
                             [.4818, 1.0893, 0.],
                             [.6818, 1.0893, 0.]], color=ANO_COLOR,
                             size=2)
    flat_hi_leader = Leader([[0., .2130, 0.], 
                             [-.1253, -.7791, 0.],
                             [-.3253, -.7791, 0.]], color=ANO_COLOR,
                             size=2)
                             
    flat_lo_text = Text("0.08", scale=0.005, size=2, pos=[.75, 1.04, 0], 
                        color=ANO_COLOR, opacity=0)
    flat_hi_text = Text("0.52", scale=0.005, size=2, pos=[-.74, -.83, 0],
                        color=ANO_COLOR, opacity=0)
    
    # Create subtraction entities
    
    meas_flat_text = Text("MEASURED FLATNESS", scale=0.005, size=2, 
                          pos=[-2.95, -.70, 0], color=WHITE, opacity=0)
    minus_text = Text("-", scale=0.005, size=2, 
                          pos=[-2.55, -.90, 0], color=ANO_COLOR, opacity=0)
    equals_text = Text("=", scale=0.005, size=2, 
                          pos=[-1.95, -.90, 0], color=ANO_COLOR, opacity=0)
    flat_dim_text = Text("0.44", scale=0.005, size=2, 
                          pos=[-1.75, -.90, 0], color=ANO_COLOR, opacity=0)
    mm_text = Text("mm", scale=0.005, size=2, 
                   pos=[-1.35, -.90, 0], color=ANO_COLOR, opacity=0)
    
    # Create dimensional entities
    
    top_dim_arrow = Leader([[-3.00, -.85, 0],
                            [-3.20, -.85, 0], 
                            [-3.20, -1.275, 0]], color=ANO_COLOR, 
                           end_arrow=True, size=1)
    bottom_dim_arrow = Leader([[-3.20, -2.00, 0], 
                               [-3.20, -1.72, 0]], color=ANO_COLOR, 
                              end_arrow=True, size=1)
    
    top_dim_line = Leader([[-3.30, -1.275, 0], 
                           [ 3.30, -1.275, 0]], color=ANO_COLOR, size=1)
    bottom_dim_line = Leader([[-3.30, -1.72, 0], 
                              [ 3.30, -1.72, 0]], color=ANO_COLOR, size=1)
    
    
    # # # Create Scene # # #
    scene = Scene(width, height, resolution, fps)
    scene.add_entities({"dial" : dial,
                        "poly" : poly,
                        "title_text_1" : title_text_1,
                        "title_text_2" : title_text_2,
                        "flat_lo_leader" : flat_lo_leader,
                        "flat_hi_leader" : flat_hi_leader,
                        "flat_lo_text" : flat_lo_text,
                        "flat_hi_text" : flat_hi_text,
                        "meas_flat_text" : meas_flat_text,
                        "minus_text" : minus_text,
                        "equals_text" : equals_text,
                        "flat_dim_text" : flat_dim_text,
                        "mm_text" : mm_text,
                        "top_dim_arrow" : top_dim_arrow,
                        "bottom_dim_arrow" : bottom_dim_arrow,
                        "top_dim_line" : top_dim_line,
                        "bottom_dim_line" : bottom_dim_line})
    
    
    # # # Animate # # #

    
    # A. Pause on initial black
    ani.pause(scene, 1.5)
    
    # B. Polygon and title text fade in
    ani.fade_in(scene, 1.5, "title_text_1", t_start=1.5)
    ani.fade_in(scene, 1.5, "title_text_2", t_start=1.5)
    ani.fade_in(scene, 1.5, "poly", t_start=1.5)
    
    # C. Title text fades out
    ani.fade_out(scene, 1.0, "title_text_1", t_start=5.5)
    ani.fade_out(scene, 1.0, "title_text_2", t_start=5.5)
    
    # D. Dial fades in
    ani.fade_in(scene, 1.5, "dial", t_start=7.0)

    # E. Poly translate left
    ani.slide(scene, 2.5, "poly", [-2.97, 0, 0], t_start=9.5)

    # F. Lower Dial
    ani.sweep_cmd(scene, 43.0, str("self.entities['dial'].track("
            "self.entities['poly'])"), t_start=12.0)
    ani.slide(scene, 2.0, "dial", [0, -.89, 0], profile="sinusoid", t_start=12)
    
    # G. Pause 
    
    # H. Poly translate right
    ani.slide(scene, 10.0, "poly", [5.94, 0, 0], profile="linear", t_start=15.0)
    
    # I. Pause
    
    # J. Poly translate left
    ani.slide(scene, 2.0, "poly", [-5.94, 0, 0], t_start=26.5)
    
    # K. Pause
    
    # L. Poly translate right with meas lines on
    ani.set_cmd(scene, "self.entities['dial'].reset_highlight()", 
            t_start=29.5)
    ani.set_cmd(scene, "self.entities['dial'].display_highlight(True)", 
            t_start=29.5)
    ani.slide(scene, 15.0, "poly", [5.94, 0, 0], profile="linear", t_start=30.)
    
    # M. Pause
    
    # N. Top meas line extends out 
    ani.sweep_attr(scene, 0.5, "flat_lo_leader", "extension", 0.0, 1.0, 
            profile="sinusoid", t_start=46.0)
    ani.fade_in(scene, 0.5, "flat_lo_text", t_start=46.5) 
    
    # O. Bottom meas line extends out
    ani.sweep_attr(scene, 0.5, "flat_hi_leader", "extension", 0.0, 1.0, 
            profile="sinusoid", t_start=48.0)
    ani.fade_in(scene, 0.5, "flat_hi_text", t_start=48.5)
    
    # P. Meas lines disappear
    ani.set_cmd(scene, "self.entities['dial'].reset_highlight()", 
            t_start=50.5)
    ani.set_cmd(scene, "self.entities['dial'].display_highlight(False)", 
            t_start=50.5)
    ani.fade_out(scene, 1.0, "flat_hi_leader", profile="linear", t_start=51.0)
    ani.fade_out(scene, 1.0, "flat_lo_leader", profile="linear", t_start=51.0)
    
    # Q. Dial moves up
    ani.slide(scene, 1.0, "dial", [0, 4, 0], t_start=52.0, profile="quadratic")
    ani.set_attr(scene, "dial", "opacity", 0, t_start=53.0)
    
    # R. Meas numbers translate over
    ani.slide_to(scene, 1.0, "flat_hi_text", [-2.95, -.9, 0], t_start=53.0)
    ani.slide_to(scene, 1.0, "flat_lo_text", [-2.35, -.9, 0], t_start=53.0)
    
    # S. Subtraction performed
    ani.fade_in(scene, 0.5, "meas_flat_text", t_start=54.0)
    ani.fade_in(scene, 0.5, "minus_text", t_start=54.0)
    ani.fade_in(scene, 0.5, "equals_text", t_start=55.5)
    ani.fade_in(scene, 0.5, "flat_dim_text", t_start=55.5)
    ani.fade_in(scene, 0.5, "mm_text", t_start=56.0)
    
    # T. Subtraction terms fade out
    ani.fade_out(scene, 1.0, "minus_text", t_start=57.5)
    ani.fade_out(scene, 1.0, "equals_text", t_start=57.5)
    ani.fade_out(scene, 1.0, "flat_lo_text", t_start=57.5)
    ani.fade_out(scene, 1.0, "flat_hi_text", t_start=57.5)
    
    # U. Result translates left
    ani.slide_to(scene, 1.0, "flat_dim_text", [-2.95, -.9, 0], t_start=58.0)
    ani.slide_to(scene, 1.0, "mm_text", [-2.55, -.9, 0], t_start=58.0)
    
    # V. Poly recenters
    ani.slide(scene, 2.0, "poly", [-2.97, 0, 0], t_start=59.0)
    
    # W. Dim lines come out
    ani.sweep_attr(scene, 0.5, "top_dim_arrow", "extension", 0.0, 1.0,
            profile="sinusoid", t_start=61.0)
    ani.sweep_attr(scene, 0.5, "bottom_dim_arrow", "extension", 0.0, 1.0,
            profile="sinusoid", t_start=61.0)
    ani.sweep_attr(scene, 0.5, "top_dim_line", "extension", 0.0, 1.0,
            profile="sinusoid", t_start=61.5)
    ani.sweep_attr(scene, 0.5, "bottom_dim_line", "extension", 0.0, 1.0,
            profile="sinusoid", t_start=61.5)
    
    # X. Fade out
    ani.fade_out(scene, 0.5, "top_dim_arrow", t_start=65.0)
    ani.fade_out(scene, 0.5, "bottom_dim_arrow", t_start=65.0)
    ani.fade_out(scene, 0.5, "top_dim_line", t_start=65.0)
    ani.fade_out(scene, 0.5, "bottom_dim_line", t_start=65.0)
    
    ani.fade_out(scene, 0.5, "meas_flat_text", t_start=65.5)
    ani.fade_out(scene, 0.5, "mm_text", t_start=65.5)
    ani.fade_out(scene, 0.5, "flat_dim_text", t_start=65.5)
    
    ani.fade_out(scene, 0.5, "poly", t_start=66.0)
    
    # Y. Final delay
    ani.pause(scene, 1.5)
    
    # # # Write Animation # # #
    
    out = cv2.VideoWriter('dial_ani.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, 
            (width, height))
    frames = scene.get_frames()
    print("\nWriting Animation....")
    for frame in frames:
        out.write(frame)
    out.release()
    print("....done\n")

    print("exiting")
    sys.exit(0)  