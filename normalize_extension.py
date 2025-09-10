import os

def rename_files_with_correct_extension(dir_path):
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):  # Make sure it's an image
                file_path = os.path.join(subdir, file)
                new_name = file.lower()  # Convert extension to lowercase
                os.rename(file_path, os.path.join(subdir, new_name))

# Apply to both 'yes' and 'no' categories
rename_files_with_correct_extension('classification_dataset/train/yes')
rename_files_with_correct_extension('classification_dataset/train/no')
rename_files_with_correct_extension('classification_dataset/valid/yes')
rename_files_with_correct_extension('classification_dataset/valid/no')

print("File extensions normalized.")
