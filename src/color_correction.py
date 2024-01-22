import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

## Application of Color Correction on the Synthically Generated Images

dirs = os.listdir("../Data/Variation_Synthetic_Generation_color_corrected/train/")
for directory in dirs:
    
    path = "../Data/Variation_Synthetic_Generation_color_corrected/train/" + directory + "/"
    files = os.listdir(path)
    
    references = []
    generated = []

    for string in files:
        underscore_count = string.count('_')
        if underscore_count == 1:
            references.append(string)
        elif underscore_count > 1:
            generated.append(string)
            
    for reference in references:
        sub_generated = [string for string in generated if reference[-8:-4] in string]
        
        print(reference)
        
        reference_img = cv2.imread(path+reference)
        
        for gen in sub_generated:
            input_img = cv2.imread(path+gen)
            
            # Convert the reference image to the LAB color space
            reference_img_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
            # Convert the input image to the LAB color space
            input_img_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
            
            # Calculate the mean and standard deviation of the reference image in each channel
            reference_img_mean, reference_img_std = cv2.meanStdDev(reference_img_lab)
            reference_img_mean = np.squeeze(reference_img_mean)
            reference_img_std = np.squeeze(reference_img_std)
            
            # Calculate the mean and standard deviation of the input image in each channel
            input_img_mean, input_img_std = cv2.meanStdDev(input_img_lab)
            input_img_mean = np.squeeze(input_img_mean)
            input_img_std = np.squeeze(input_img_std)
            
            # Apply color correction to the input image using the mean and standard deviation of the reference image and input image in each channel
            for i in range(3):
                input_img_lab[:, :, i] = ((input_img_lab[:, :, i] - input_img_mean[i]) * (reference_img_std[i] / input_img_std[i])) + reference_img_mean[i]
                
            # Convert the color-corrected image back to the BGR color space
            output_img = cv2.cvtColor(input_img_lab, cv2.COLOR_LAB2BGR)
            
            cv2.imwrite(path + gen, output_img)