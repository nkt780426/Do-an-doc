import os
import cv2
from tqdm import tqdm
import shutil
import traceback
import re

path_folder = r'last_data'
root_folder = r'Photoface_dist\PhotofaceDB'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
cnt = 0
for id in tqdm(list_folder_id):
    try:
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            with open(os.path.join(root_folder, path_folder_id_time[10:], 'metadataII.txt'), 'r') as f:
                data = f.read()
                occlusion_check = data.split(',')[5]
                if occlusion_check != ' ':
                    print(occlusion_check, data)
                    shutil.rmtree(path_folder_id_time)
    except:
        print(id)
        traceback.print_exc()

def find_empty_subdirectories(parent_dir):
    empty_subdirectories = []

    # Lặp qua tất cả các thư mục con trong thư mục cha
    for root, dirs, files in os.walk(parent_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            # Kiểm tra xem thư mục con có chứa tệp không
            if not os.listdir(dir_path):
                empty_subdirectories.append(dir_path)

    return empty_subdirectories

# Đường dẫn đến thư mục cha
parent_directory = 'last_data'

# Tìm và hiển thị các thư mục con không có tệp
empty_directories = find_empty_subdirectories(parent_directory)
if empty_directories:
    print("Các thư mục con không có tệp:")
    for directory in empty_directories:
        shutil.rmtree(directory)
        print(directory)
else:
    print("Tất cả các thư mục con đều có tệp.")
