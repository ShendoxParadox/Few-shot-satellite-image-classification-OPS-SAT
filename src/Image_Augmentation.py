# %%
from skimage import transform
import imageio as iio
import cv2
import os
import Augmentor
import albumentations as A
from imgaug import augmenters as iaa

## Rotation

# %% rotation angles
rotations = [90]
dirs = os.listdir("../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/")
for directory in dirs:
    
    path = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/"
    files = os.listdir(path)
    
    
    for file in files:
        
        img = cv2.imread(path + file)
        
        for r in rotations:
            rotated_img = transform.rotate(img,
                                  angle = r,
                                  resize=False)
        
            file_write = file[:-4]
            rotated_img = cv2.convertScaleAbs(rotated_img, alpha=(255.0))
            cv2.imwrite("../Data/Variation_Synthetic_Generation_full_filtered_augmented//train/" + directory + "/" + file_write + "_" + "rotated_" + str(r) + ".png", rotated_img)



# %%
## Flip            
for directory in dirs:
    
    path = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/"
    files = os.listdir(path)
#     n_samples = len(files)
    
    p = Augmentor.Pipeline(path)
    # p.flip_left_right(probability=1)
    # p.process()
    p.flip_top_bottom(probability=1)
    p.process()
    
    

# %%
## Brightness and Contrast  
for directory in dirs:
    
    path = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/"
    files = os.listdir(path)
    
    new_folder = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/" + "brightness_contrast"
    # Create the folder
    os.makedirs(new_folder, exist_ok=True)


    for file in files:
        
        img = cv2.imread(path + file)
        
            
        transform = A.Compose([A.RandomBrightnessContrast(p=1, brightness_by_max=False)])
        transformed = transform(image = img)
        transformed_img = transformed["image"]


        file_write = file[:-4]
        cv2.imwrite("../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/brightness_contrast/" + file_write + "_bright_contrast_" + ".png", transformed_img)




# %%
## Noise 
for directory in dirs:
    
    path = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/"
    files = os.listdir(path)
    
    new_folder = "../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/" + "noise"
    # Create the folder
    os.makedirs(new_folder, exist_ok=True)
    
    for file in files:
        
        img = iio.imread(path + file)
        
            
        transform = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=(1,3))])
        transformed_img = transform(image = img)

        file_write = file[:-4]
        cv2.imwrite("../Data/Variation_Synthetic_Generation_full_filtered_augmented/train/" + directory + "/noise/" + file_write + "_" + "noise" + ".png", cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
# %%
