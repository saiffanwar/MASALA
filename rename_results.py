import os

# Define the parent directory
parent_dir = 'saved/feature_ensembles'

# Define old and new filenames
old_filename = '1_0.05_0.05.pck'
new_filename = '1_0.02_5.pck'

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(parent_dir):
    if old_filename in filenames:
        old_path = os.path.join(dirpath, old_filename)
        new_path = os.path.join(dirpath, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
