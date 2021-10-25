import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import math
import sys
from skimage.metrics import structural_similarity as ssim
np.set_printoptions(threshold=sys.maxsize)


def main():
    for i in range(57):
        os.mkdir('./dataset/feature_map_layer1/'+str(i))
    




if __name__ == '__main__':
	main()

