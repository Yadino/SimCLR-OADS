import os
import random
import shutil

"""
Create a new dataset from a subset of the images of each class. 
The requirement for the dataset is to have a directory for each class.
"""

# Path to the root folder containing class folders
root_folder = r"C:\Users\YO\OneDrive - UvA\ML_new"
dest_folder = r"D:\01 Files\04 University\00 Internships and theses\2. AI internship\Practice\ML_subset"

# Percentage of samples to select (5%)
percentage = 5

# Iterate through each class folder
for class_folder in os.listdir(root_folder):
    class_path = os.path.join(root_folder, class_folder)

    # List all files/samples in the class folder
    files = os.listdir(class_path)

    # Calculate 5% of the total number of samples
    num_samples_to_select = int(len(files) * (percentage / 100))

    # Randomly select samples
    selected_samples = random.sample(files, num_samples_to_select)

    # Iterate through selected samples
    for sample in selected_samples:
        source_path = os.path.join(class_path, sample)

        # Replicate folder structure and copy selected samples
        destination_folder = os.path.join(dest_folder, class_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        shutil.copy(source_path, destination_folder)