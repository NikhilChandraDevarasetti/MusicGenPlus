import os

def read_text_files_in_folder(folder_path):
    descriptions = []
    filenames = []
    
    # Iterate over all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a text file and not a directory
        if filename.endswith(".txt") and not os.path.isdir(os.path.join(folder_path, filename)):
            filenames.append(filename.split('.txt')[0])  # Add only .txt filenames to the list
            file_path = os.path.join(folder_path, filename)
            
            # Open and read the text file
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                descriptions.append(text)

    return descriptions, filenames

# Specify the folder path
folder_path = '/home/nikhil/Documents/yeshiva_projects/audiocraft/musiccaps_training_data'


# Prompts
# 'This is a nice guitar music', 
descriptions, filenames = read_text_files_in_folder(folder_path)


# Write to file
with open('descriptions.txt', 'w') as file:
    file.write(str(descriptions))
    
with open('filenames.txt', 'w') as file:
    file.write(str(filenames))
