import cv2
import os
import numpy as np
import pickle



def main():
    cls = 8
    train_data = []
    train_label = []
    for c in range(cls):
        f = 0
        train_img = []
        train_path = "./dataset/gt_srcnnmodel_8cls_pickle/train/0"+str(c)+"/"
        for filename in os.listdir(train_path):
            with open("./dataset/gt_srcnnmodel_8cls_pickle/train/0"+str(c)+"/"+filename, "rb") as f_in:
                tmp_input = pickle.load(f_in)
            
            if train_img is not None:
                train_img.append(tmp_input)
        label_tmp = np.full((400,1),c)
        
        if f != 0:
            train_data = np.append(train_data,train_img,axis=0)
        else:
            train_data = train_img
        if f != 0:
            train_label = np.append(train_label,label_tmp,axis=0)
        else:
            train_label = label_tmp
        f = 1


    train_label = np.squeeze(train_label)
    train_label = np.array(train_label)
    train_data = np.array(train_data)
    train_data = train_data.astype('float32')
    train_label = train_label.astype('float32')
    print(train_data)
    print(train_label.shape)

if __name__ == '__main__':
	main()

