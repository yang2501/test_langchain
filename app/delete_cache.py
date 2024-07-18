import os
import shutil

def delete_cache_folder():
    """
    Deletes the cache folder if it exists.
    
    Parameters:
    cache_folder_path (str): Path to the cache folder.
    """
    cache_folder_path = './cache'  # Change this path to your cache folder path
    # chromadb_folder_path = "./chromadb"
    paths_to_folders_to_delete = [cache_folder_path]
    
    for path in paths_to_folders_to_delete:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Cache folder '{path}' has been deleted.")
            else:
                print(f"'{path}' exists but is not a folder.")
        else:
            print(f"Cache folder '{path}' does not exist.")
    




