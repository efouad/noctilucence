#test.py
import numpy as np
import cv2


def modify():
    val = np.array([[1, 2, 3], [4, 5, 6]])
    return val

if __name__ == '__main__':
    the_val = modify()
    print(the_val)
    modify() = np.array([[10000, 2, 3], [4, 5, 6]])