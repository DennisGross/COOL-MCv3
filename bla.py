import os

def check_folder_in_tree(folder_name, root_folder):
    for root, dirs, files in os.walk(root_folder):
        if folder_name in dirs:
            print(root)
            print(dirs)
            return True
    return False



exists = check_folder_in_tree("9326488d9f324c7cb9ca57525a346c50", "mlruns")
print(exists)
