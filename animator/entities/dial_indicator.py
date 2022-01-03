#
# entities.dial_indicator.py
#
# Simple dial indicator Entity object. 
#


import numpy as np
from numpy.linalg import inv
from pyquaternion import Quaternion
import cv2
from collections.abc import Iterable
from entities.entity import Entity
from entities.primatives import *
import copy
from funcs import *


class Dial_Indicator(Entity):
    """ A manual a dial indicator with gauge and needle. 
    
    Attributes:
        deflection:
            A float in mm representing the distance the gauge tip has depressed.        
        readout: 
            A float from 0 to 100, representing the needle readout.
            Needle is 12 o'clock at value 0, and increases clockwise.
        diameter:
            A float in mm representing the diameter of dial indicator that the
            needle will fit inside.
        needle:
            Entity representing the pointed indicator
        dial:
            Entity representing the static dial and gauge
        highlight_show:
            Boolean to indicate whether min & max lines & sector highlight are
            visible to indicate the sweep in needle position
        plunger_show: 
            Boolean to indicate whether plunger is visible on gauge
        readout_scale: 
            Plunger deflection (mm) corresponding to 1 full dial revolution 
            (readout = 100)
    """
    # Needle
    ARROW_LG_SCALE = .455 # Ratio of needle length to dial diameter
    ARROW_WD_SCALE = .020 # Ratio of arrow width (at center cap) to dial dia
    CENTER_CAP_SCALE = .070 # Ratio of center cap diameter to dial diameter

    
    # Rim
    RIM_THICK_SCALE = .95 # Ratio of face diameter to full dial diameter
    MEAS_CIR_SCALE = .80 # Ratio of measurement circle radius to dial diameter
    HOLDER_WD_SCALE = .130 # Ratio of holder width to dial diameter
    HOLDER_LG_SCALE = .49 # Ratio bottom radial location of holder to dial dia
    COLUMN_WD_SCALE = .133 # Ratio of holder width to dial diameter
    COLUMN_LOW_LG_SCALE = .796 # Ratio of bottom edge of column to dial dia
    COLUMN_LOW_CHM_LG_SCALE = .7 # Ratio center <-> bottom chamfer to dial dia
    COLUMN_HGH_LG_SCALE = .547 # Ratio of top edge of column to dial dia
    COLUMN_HGH_CHM_LG_SCALE = .531 # Ratio of center <-> top chamfer to dial dia
    COLUMN_CHM_WD_SCALE = .101 # Ratio of width of chamfer rim on either end
    
    # Ticks
    MAJ_TICK_SCALE = .80 # Ratio of dia where major ticks begin to dial dia 
    MED_TICK_SCALE = .84 # Ratio of dia where medium ticks begin to dial dia 
    MIN_TICK_SCALE = .88 # Ratio of dia where minor ticks begin to dial dia 
    
    # Number
    NUM_SCALE = .0018723 # Ratio of number font scale factor to dial dia
    NUM_POS_SCALE = .35 # Ratio of number radial position to dial dia
    NUM_SHIFT_SCALE = [-0.035, -.02] # Ratio of num position shift to dial dia
    ZERO_SHIFT_SCALE = .0175 # Ratio of 0 pos shift (since 1 digit) to dial dia
    
    # Plunger
    PLUNG_DIA_SCALE = 0.051 # Ratio of main plunger dia to dial dia
    PLUNG_TOP_DIA_SCALE = .101 # Ratio of top stop dia to dial dia
    PLUNG_TOP_CHM_DIA_SCALE = .05 # Ratio of top chamfer dia to dial dia
    PLUNG_TIP_MOUNT_DIA_SCALE = .063 # Ratio of plung. tip mount dia to dial dia
    PLUNG_TIP_MOUNT_CHM_DIA_SCALE = .005 # Ratio of plunger tip dia to dial dia
    
    PLUNG_TOP_LG_SCALE = .646 # Ratio of center <-> top to dial dia
    PLUNG_TOP_CHM_LG_SCALE = .637 # Ratio of center <-> top chm to dial dia
    PLUNG_LG_SCALE = 1.535 # Ratio of center <-> end of plunger body to dial dia
    PLUNG_TIP_LG_SCALE = 1.595 # Ratio ctr <-> tip to dial dia
    PLUNG_TIP_CHM_LG_SCALE = 1.545 # Ratio ctr <-> tip mount chm to dial dia
    
    # Tracking distance
    PLUNG_TIP_TRACK_SCALE = 1.610 # Ratio of ctr <-> probe dist to dial dia
    
    # Legend
    LEGEND_ARROW_SCALE = .03 # Ratio of legend arrow length to dial dia
    
    # Leader line distances in mm, unscaled:
    LEGEND_L1A_START = np.array([-1.2, 0, 0])
    LEGEND_L1A_END = np.array([-.25, 0, 0])
    LEGEND_L1B_START = np.array([-.25, .375, 0])
    LEGEND_L1B_END = np.array([-.25, -.375, 0])
    
    LEGEND_L2A_START = [-1, 1, 1] * LEGEND_L1A_START
    LEGEND_L2A_END = [-1, 1, 1] * LEGEND_L1A_END
    LEGEND_L2B_START = [-1, 1, 1] * LEGEND_L1B_START
    LEGEND_L2B_END = [-1, 1, 1] * LEGEND_L1B_END
    
    # Arrow triangle distances in mm, unscaled:
    LEGEND_T1_P1 = np.array([-.625, .1875, 0])
    LEGEND_T1_P2 = np.array([-.25, 0, 0])
    LEGEND_T1_P3 = np.array([-.625, -.1875, 0])
    
    LEGEND_T2_P1 = [-1, 1, 1] * LEGEND_T1_P1
    LEGEND_T2_P2 = [-1, 1, 1] * LEGEND_T1_P2
    LEGEND_T2_P3 = [-1, 1, 1] * LEGEND_T1_P3

    LEGEND_NUM_SCALE = .001 # Ratio of number font scale factor to dial dia
    LEGEND_DISP_SCALE = np.array([-.07, -.18, 0]) # Pos ratios to dial dia
    LEGEND_TXT_OFFSET = np.array([.3, -.375, 0]) # Text disp. (mm), unscaled 
    
    NEEDLE_COLOR = [0, 0, 128] # Maroon needle
    FACE_COLOR = [210, 210, 210] # Light gray gauge face
    RIM_COLOR = [70, 70, 70] # Dark gray rim
    MEAS_WEDGE_COLOR = [234, 217, 153] # Light blue wedge
    MEAS_WEDGE_OPACITY = .30
    MEAS_LINE_COLOR = [213, 90, 0] # dark blue lines
    MEAS_LINE_OPACITY = .60
    PLUNG_COLOR = [190, 190, 190] # Light gray plunger 
    PLUNG_TIP_COLOR = [0, 0, 128] # Plunger tip
    
    
    def __init__(self, diameter, deflection=0, highlight_show=False, 
                    plunger_show=True, readout_scale=1.00, **kwargs):
        """ Creates a new dial indicator. 
        
        Args:
            diameter: (float) Diameter of dial indicator to fit inside (mm) 
            deflection: (float) Gauge plunger depression (mm)
            highlight_show: (boolean) Whether swept value highlight is visible 
            plunger_show: (boolean) Whether plunger should be visible
            readout_scale: (float) Plunger distance (mm) corresponding to 1
                           full dial revolution (readout = 100)
        """
        super().__init__(**kwargs)
        self.attributes["diameter"] = diameter
        self.attributes["readout_scale"] = readout_scale
        
        # ## Needle ## #
        center_cap = Disk(diameter * self.CENTER_CAP_SCALE / 2.0, 
                color=self.NEEDLE_COLOR)
        arrow = Polygon([P([0, diameter * self.ARROW_LG_SCALE, 0]), 
                         P([diameter * self.ARROW_WD_SCALE / 2.0, 0, 0]), 
                         P([-diameter * self.ARROW_WD_SCALE / 2.0, 0, 0])],
                         color=self.NEEDLE_COLOR)
        self.needle = Entity(components=[center_cap, arrow], 
                color=self.NEEDLE_COLOR)
        # ## Dial ## #
        self.dial = Entity()
        
        # Rim & Holder #
        self.rim = Disk(diameter / 2.0, size=5, color=self.RIM_COLOR)
        self.holder_circle = Disk(diameter * self.HOLDER_WD_SCALE / 2., 
                               pos=[0, -diameter * self.HOLDER_LG_SCALE, 0],
                               color=self.RIM_COLOR)
        self.holder_column = Polygon([
                P([-diameter * self.COLUMN_WD_SCALE / 2., 
                    diameter * self.COLUMN_HGH_CHM_LG_SCALE, 0]),
                P([-diameter * self.COLUMN_CHM_WD_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([ diameter * self.COLUMN_CHM_WD_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([ diameter * self.COLUMN_WD_SCALE / 2., 
                    diameter * self.COLUMN_HGH_CHM_LG_SCALE, 0]),
                            
                P([ diameter * self.COLUMN_WD_SCALE / 2., 
                   -diameter * self.COLUMN_LOW_CHM_LG_SCALE, 0]),
                P([ diameter * self.COLUMN_CHM_WD_SCALE / 2., 
                   -diameter * self.COLUMN_LOW_LG_SCALE, 0]),
                P([-diameter * self.COLUMN_CHM_WD_SCALE / 2., 
                   -diameter * self.COLUMN_LOW_LG_SCALE, 0]),
                P([-diameter * self.COLUMN_WD_SCALE / 2., 
                   -diameter * self.COLUMN_LOW_CHM_LG_SCALE, 0])],
                color=self.RIM_COLOR)
        
        self.dial.add_components(self.rim, self.holder_column, 
                self.holder_circle)
        
        # Plunger # 
        self.plunger_body = Polygon([
                P([-diameter * self.PLUNG_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_DIA_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_TOP_DIA_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_TOP_DIA_SCALE / 2., 
                    diameter * self.PLUNG_TOP_CHM_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_TOP_CHM_DIA_SCALE / 2., 
                    diameter * self.PLUNG_TOP_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TOP_CHM_DIA_SCALE / 2., 
                    diameter * self.PLUNG_TOP_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TOP_DIA_SCALE / 2., 
                    diameter * self.PLUNG_TOP_CHM_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TOP_DIA_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_DIA_SCALE / 2., 
                    diameter * self.COLUMN_HGH_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_LG_SCALE, 0])],
                color=self.PLUNG_COLOR)
        
        self.plunger_tip = Polygon([
                P([-diameter * self.PLUNG_TIP_MOUNT_CHM_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_TIP_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_TIP_MOUNT_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_TIP_CHM_LG_SCALE, 0]),
                P([-diameter * self.PLUNG_TIP_MOUNT_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TIP_MOUNT_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TIP_MOUNT_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_TIP_CHM_LG_SCALE, 0]),
                P([ diameter * self.PLUNG_TIP_MOUNT_CHM_DIA_SCALE / 2., 
                   -diameter * self.PLUNG_TIP_LG_SCALE, 0])], 
                color=self.PLUNG_TIP_COLOR)
                   
        self.plunger = Entity(components=[self.plunger_body, 
                self.plunger_tip])

                       
        
        # Face #
        face = Disk(diameter / 2.0 * self.RIM_THICK_SCALE, 
                  color=self.FACE_COLOR)
        self.dial.add_components(face)
        
        # Ticks # 
        indices = 100
        for i in range(indices):
            if i % 10 == 0: # major index
                tick = Line_Seg([0, diameter * self.MAJ_TICK_SCALE / 2.0, 0], 
                            [0, diameter * (1 + self.RIM_THICK_SCALE) / 4.0, 0],
                            size=1, color=self.RIM_COLOR)
            elif i % 5 == 0: # medium index
                tick = Line_Seg([0, diameter * self.MED_TICK_SCALE / 2.0, 0], 
                            [0, diameter * (1 + self.RIM_THICK_SCALE) / 4.0, 0],
                            size=1, color=self.RIM_COLOR)
            else:
                tick = Line_Seg([0, diameter * self.MIN_TICK_SCALE / 2.0, 0], 
                            [0, diameter * (1 + self.RIM_THICK_SCALE) / 4.0, 0],
                            size=1, color=self.RIM_COLOR)
            tick.move(dori=Quaternion(axis=[0, 0, 1], 
                                      radians = i / indices * 2 * np.pi))
            self.dial.add_components(tick)
        
        # Numbers #
        for i in range(0, indices, 10):
            self.dial.add_components(Text(str(i), 
                     self.NUM_SCALE * diameter, 
                     size=1, 
                     color=self.RIM_COLOR, 
                     pos = [self.NUM_SHIFT_SCALE[0] * diameter + 
                                self.NUM_POS_SCALE * diameter * 
                                np.cos(-2 * np.pi * i / indices + np.pi / 2) +
                                self.ZERO_SHIFT_SCALE * diameter * (i == 0),                                
                            self.NUM_SHIFT_SCALE[1] * diameter + 
                                self.NUM_POS_SCALE * diameter * 
                                np.sin(-2 * np.pi * i / indices + np.pi / 2), 
                            0]))
        
        # Measurement Wedge, min & max lines #
        self.meas_wedge = Wedge(self.RIM_THICK_SCALE * diameter / 2., 
                           np.pi/2, np.pi/2,
                           color=self.MEAS_WEDGE_COLOR, 
                           opacity=self.MEAS_WEDGE_OPACITY)
        self.meas_min_line = Line_Seg([0, 0, 0], 
                                 [0, self.RIM_THICK_SCALE * diameter / 2., 0],
                                 size=2, color=self.MEAS_LINE_COLOR,
                                 opacity=self.MEAS_LINE_OPACITY)
        self.meas_max_line = Line_Seg([0, 0, 0], 
                                 [0, self.RIM_THICK_SCALE * diameter / 2., 0],
                                 size=2, color=self.MEAS_LINE_COLOR,
                                 opacity=self.MEAS_LINE_OPACITY)
        
        self.dial.add_components(self.meas_wedge, self.meas_min_line, 
                self.meas_max_line)
        
        # Legend #
        arrow_scale = self.LEGEND_ARROW_SCALE * diameter
        
        self.legend_leader_1a = Line_Seg(arrow_scale * self.LEGEND_L1A_START, 
                                         arrow_scale * self.LEGEND_L1A_END,
                                         color=self.RIM_COLOR)
        self.legend_leader_1b = Line_Seg(arrow_scale * self.LEGEND_L1B_START, 
                                         arrow_scale * self.LEGEND_L1B_END,
                                         color=self.RIM_COLOR)
        self.legend_leader_2a = Line_Seg(arrow_scale * self.LEGEND_L2A_START, 
                                         arrow_scale * self.LEGEND_L2A_END,
                                         color=self.RIM_COLOR)
        self.legend_leader_2b = Line_Seg(arrow_scale * self.LEGEND_L2B_START, 
                                         arrow_scale * self.LEGEND_L2B_END,
                                         color=self.RIM_COLOR)
        self.legend_triangle_1 = Polygon([P(arrow_scale * self.LEGEND_T1_P1),
                                          P(arrow_scale * self.LEGEND_T1_P2),
                                          P(arrow_scale * self.LEGEND_T1_P3)],
                                         color=self.RIM_COLOR)
        self.legend_triangle_2 = Polygon([P(arrow_scale * self.LEGEND_T2_P1),
                                          P(arrow_scale * self.LEGEND_T2_P2),
                                          P(arrow_scale * self.LEGEND_T2_P3)],
                                         color=self.RIM_COLOR)

        self.legend_text = Text("  0.01mm", self.LEGEND_NUM_SCALE * diameter, 
                size=1, color=self.RIM_COLOR, 
                pos=arrow_scale * self.LEGEND_TXT_OFFSET)
                
        self.legend = Entity(components=[self.legend_leader_1a,
                self.legend_leader_1b, self.legend_leader_2a, 
                self.legend_leader_2b, self.legend_triangle_1, 
                self.legend_triangle_2, self.legend_text])
        self.legend.move(dpos=diameter * self.LEGEND_DISP_SCALE)
        self.dial.add_components(self.legend)

        
        # Combine dial, needle and plunger
        self.add_components(self.plunger, self.dial, self.needle)
        
        # Set deflection.
        self.set_deflection(deflection)
        self.reset_highlight() # Reset highlight to the deflection specified
        self.display_highlight(highlight_show)
        self.update_plunger()
        self.display_plunger(plunger_show)
    
    def set_deflection(self, deflection):
        """ Sets new gauge deflection, moves plunger, and updates dial readout.
        
        Args:
            deflection: (float) New plunger depression (mm)
        """
        self.attributes["deflection"] = deflection
        # Adjust plunger position.
        self.update_plunger()
        # Adjust dial
        readout = (deflection / self.attributes["readout_scale"] * 100) % 100
        self.set_readout(readout)
        
    def set_readout(self, readout):
        """ Changes the position on the dial. 
        
        Args:
            readout: Float from 0 - 100. Dial position, with 0 at 12 o'clock
                        and 25 at 3 o'clock.
        """
        
        # Needle #
        self.attributes["readout"] = readout
        angle = -readout * 2. * np.pi / 100.
        self.needle.move(ori=np.array([[np.cos(angle), -np.sin(angle), 0], 
                                       [np.sin(angle), np.cos(angle), 0], 
                                       [0, 0, 1]]))
        # Min/Max Highlight #
        if "min_swept" in self.attributes and "max_swept" in self.attributes:
            if self.check_min_max() == -1: # min exceedance
                self.attributes["min_swept"] = readout
            elif self.check_min_max() == 1: # max exceedance 
                self.attributes["max_swept"] = readout
        else: # Not established; on initial object creation
            self.reset_highlight()
        self.set_min_max_swept()
        

                                       
    def check_min_max(self):
        """ Checks if current needle readout is within historical min/max.

        Returns:
            int: 0 if needle in bounds. -1 if exceeds min. 1 if exceeds max.
        """
        
        # Calc angles. #
        readout_angle = (-self.attributes["readout"] * 2. * np.pi / 100.)
        min_angle = -self.attributes["min_swept"] * 2. * np.pi / 100.
        max_angle = -self.attributes["max_swept"] * 2. * np.pi / 100.
        
        # Create unit vectors. #
        readout_vector = np.array([np.cos(readout_angle), 
                                      np.sin(readout_angle), 0])
        min_vector = np.array([np.cos(min_angle), 
                               np.sin(min_angle), 0])
        max_vector = np.array([np.cos(max_angle), 
                               np.sin(max_angle), 0])
        
        # Calculate cross product. Negative cross product indicates exceedance.
        min_exceeded = np.sign(np.cross(min_vector, readout_vector))[2] <= 0
        max_exceeded = np.sign(np.cross(readout_vector, max_vector))[2] <= 0
        
        # Check which direction is more closely exceeded.
        if min_exceeded and not max_exceeded:
            return -1
        if max_exceeded and not min_exceeded:
            return 1
        if min_exceeded or max_exceeded:
            if (np.dot(min_vector, readout_vector) >= 
                    np.dot(max_vector, readout_vector)):
                return -1
            else:
                return 1
        
        # If no exceedance,
        return 0
    
    def reset_highlight(self):
        """ Resets the min/max swept values, clearing the highlight if visible.
        """
        self.attributes["min_swept"] = self.attributes["readout"]
        self.attributes["max_swept"] = self.attributes["readout"]
        self.set_min_max_swept()
        
    def set_min_max_swept(self):
        """ Updates min & max swept lines according to current attribute values.
        """
        
        # Update min sweep line:
        min_ang = -self.attributes["min_swept"] * 2. * np.pi / 100.
        self.meas_min_line.move(
            ori=np.array([[np.cos(min_ang), -np.sin(min_ang), 0], 
                          [np.sin(min_ang), np.cos(min_ang), 0], 
                          [0, 0, 1]]))
        
        # Update max sweep line:
        max_ang = -self.attributes["max_swept"] * 2. * np.pi / 100.
        self.meas_max_line.move(
            ori=np.array([[np.cos(max_ang), -np.sin(max_ang), 0], 
                          [np.sin(max_ang), np.cos(max_ang), 0], 
                          [0, 0, 1]]))
        
        # Update swept wedge:
        self.meas_wedge.attributes["start_ang"] = np.pi / 2 + min_ang
        self.meas_wedge.attributes["end_ang"] = np.pi / 2 + max_ang
        
    def display_highlight(self, highlight_show):
        """ Enables or disables min/max swept highlight from appearing on gauge.
        
        Args:
            highlight_show: (boolean) If min/max highlighting should be visible.
        """
        self.attributes["highlight_show"] = highlight_show
        if highlight_show:
            self.meas_wedge.attributes["visible"] = True
            self.meas_min_line.attributes["visible"] = True
            self.meas_max_line.attributes["visible"] = True
        else:
            self.meas_wedge.attributes["visible"] = False
            self.meas_min_line.attributes["visible"] = False
            self.meas_max_line.attributes["visible"] = False
    
    def update_plunger(self):
        """ Updates the plunger position based on the current gauge deflection.
        """
        self.plunger.move(pos=[0, self.attributes["deflection"], 0])
    
    def display_plunger(self, plunger_show):
        """ Enables or disables plunger visibility.
        
        Args:
            plunger_show: Boolean indicating whether plunger should be visible.
        """
        self.attributes["plunger_show"] = plunger_show
        self.plunger.attributes["visible"] = plunger_show
        
    def track(self, poly):
        #TODO: Work on more than just polygons
        #TODO: Implement rotation of polygon
        #TODO: Work in 3d
        """ Sets the deflection of gauge to track the profile of a polygon.
        Uses positions and orientation of this indicator and polygon. 
        If no intersection, sets dial to max extended length. 
        
        Args:
            poly  (Polygon)  Entity to track
        """
        
        free_lg = self.PLUNG_TIP_TRACK_SCALE * self.attributes["diameter"]
        
        # Dial starts pointing straight down, rotate by global orientation
        dir = np.dot(self.gori(), np.array([0, -1, 0])) 
        
        dist = abs(dist_to_poly(poly, P(self.gpos()), dir))
        self.set_deflection(max(free_lg - dist, 0))
        
        
        