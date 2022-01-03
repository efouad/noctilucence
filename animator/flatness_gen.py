# Flatness Gen
# Generates a random uneven flatness surface through random walk.
#

# 1D implementation

from random import random as rand
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    """ Returns height vs pos 1d flatness data according to constants specified.
    """
    LENGTH = 6 # mm; length of curve to be made uneven
    LEN_INCREMENT = .05 # mm; resolution of distance measurements along curve 
    MAX_DELTA = .04 # mm; maximum change in height in each length increment
    START_HEIGHT = .15 # mm; height at beginning of curve
    MIN_HEIGHT = .04 # mm; height does not fall below
    MAX_HEIGHT = .62 # mm; height does not rise above
    
    pos = 0
    height = START_HEIGHT
    points = [[pos, height]] # list of [pos, height] with pos from 0 to LENGTH
    
    while pos < LENGTH:
        height_delta = (2 * rand() - 1) * MAX_DELTA
        if height + height_delta > MAX_HEIGHT:
            height = MAX_HEIGHT
        elif height + height_delta < MIN_HEIGHT:
            height = MIN_HEIGHT
        else:
            height += height_delta
        points.append([pos, height])
        pos += LEN_INCREMENT
    return np.array(points)
    

if __name__ == '__main__':
    FLATNESS_TARGET = .44
    # Get data
    while True:
        points = get_data()
        # Round min and max to nearest values
        points[np.argmax(points, 0)[1], 1] = np.round(
                points[np.argmax(points, 0)[1], 1], 2)
        points[np.argmin(points, 0)[1], 1] = np.round(
                points[np.argmin(points, 0)[1], 1], 2)
        # Check flatness meets flatness target
        flatness = np.max(points[:,1]) - np.min(points[:,1])
        if abs(flatness - FLATNESS_TARGET) < .001:
            break
    
    
    # Write file
    with open("flatness_vals.csv", "w") as output:
        for point in points:
            output.write(str(point[0]) + "," + str(point[1]) + "\n")
            
    # Create plot
    plt.plot(points[:,0], points[:,1])
    plt.show()