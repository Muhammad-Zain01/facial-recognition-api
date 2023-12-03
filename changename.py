import os

# Set the directory path where the files are located
directory = 'temp3/'

# Loop through the files in the directory
i = 0
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):  # Specify the file extension you want to rename
        
        newFile = "_1117_"+str(i)+".jpg"
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, newFile)
        os.rename(old_filepath, new_filepath)
        
        print(f"Renamed {filename} to {newFile}")
        i += 1
