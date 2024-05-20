import os

def add_prefix_to_images(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate over each file
    for idx, file_name in enumerate(file_list):
        # Check if the file is an image file
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Construct the new file name with the 'n' prefix
            new_file_name = '[dog2fubao]output_' + str(idx + 1) + '.png'

            # Rename the file with the new file name
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

    print("Prefix added to image file names.")

    # Remove the original image files
    for file_name in file_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            os.remove(os.path.join(folder_path, file_name))

    print("Original image files removed.")

# Call the function with the folder path where the images are located
add_prefix_to_images('./temp')

