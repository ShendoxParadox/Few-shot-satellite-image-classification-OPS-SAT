# %% 
import os
from PIL import Image

def crop_and_resize_images(directory, output_size=(200, 200)):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' not found.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if the file is an image
        if file_path.lower().endswith(('.png')):
            # Open the image
            with Image.open(file_path) as img:
                # Crop the image's borders, assuming we want to remove 10% of each side
                width, height = img.size
                left = width * 0.01
                top = height * 0.01
                right = width * 0.99
                bottom = height * 0.9
                cropped_img = img.crop((left, top, right, bottom))

                # Resize the image
                resized_img = cropped_img.resize(output_size, Image.ANTIALIAS)

                # Save the image back to the same location
                resized_img.save(file_path)
                print(f"Processed and saved {filename}")

# %%
directory = '../Data/generate_2/process'  # Change this to the path of your image folder
crop_and_resize_images(directory)

# %%
