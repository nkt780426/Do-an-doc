import os
import cv2
from tqdm import tqdm
import shutil
import traceback
import re

path_folder = r'final_data'
root_folder = r'Photoface_dist\PhotofaceDB'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
data_ok = []
cnt = 0
for id in tqdm(list_folder_id):
    try:
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            with open(os.path.join(root_folder, path_folder_id_time[11:], 'LightSource.m'), 'r') as f:
                data = f.read()
                numbers = re.findall(r'\d+\.\d+|\d+', data)
                new_data = ''
                for number in numbers[1::2]:
                    new_data += number + ' '
                with open(os.path.join(path_folder_id_time, 'LightSource.txt'), 'w') as f:
                    f.write(new_data[:-1])
    except:
        print(id)
        traceback.print_exc()