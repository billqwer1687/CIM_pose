import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt 
import math
from skimage.metrics import structural_similarity as ssim



def main():
    cls = 8
    train_path = "./dataset/ccf_output/train/04/200.pickle"

    with open(train_path, "rb") as f_in:
        tmp_input = pickle.load(f_in)
        tmp_input = tmp_input.flatten()
        tmp_input = np.where(tmp_input < 0, tmp_input * (-63/tmp_input.min()), tmp_input * (63/tmp_input.max()))
        tmp_input = tmp_input * 2
        tmp_input = np.where(tmp_input < -63, -63, tmp_input)
        tmp_input = np.where(tmp_input > 63, 63, tmp_input)
        tmp_input = np.round(tmp_input)
        plt.hist(tmp_input)
        plt.savefig("test.png")
            # tmp_input = np.where(tmp_input < 0, tmp_input * (-63/tmp_input.min()), tmp_input * (63/tmp_input.max()))
                # tmp_input = tmp_input * factor
                # tmp_input = np.where(tmp_input < -63, -63, tmp_input)
                # tmp_input = np.where(tmp_input > 63, 63, tmp_input)
                # tmp_input = np.round(tmp_input)
    




if __name__ == '__main__':
	main()

