import os
import cv2
from tqdm import tqdm
import shutil
import traceback

path_folder = r'cropped_data'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
data_ok = []
cnt = 0
for id in tqdm(list_folder_id):
    check_ok1 = False
    try:
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            list_name_image = [x[:-4].split('_')[1:] for x in os.listdir(path_folder_id_time) if x.endswith(".png")]
            check_ok = True
            for data in list_name_image[1:]:
                if int(data[2]) < 100:
                    check_ok = False
                    break
            if check_ok:
                check_ok1 = True
                data_ok.append(path_folder_id_time)
                # cnt += 1
    except:
        print(id)
        traceback.print_exc()
    if check_ok1:
        cnt += 1

print(cnt)
# print((data_ok))

print(os.path.join('final_data', data_ok[0][13:]))
for tmp in data_ok:
    try:
        shutil.copytree(tmp, os.path.join('final_data', tmp[13:]))
    except:
        traceback.print_exc()

path_folder = r'last_data'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
cnt = 0
for id in tqdm(list_folder_id):
    path_folder_id = os.path.join(path_folder, id)
    list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]
    for time_id in list_folder_id_time:
        cnt += 0
    cnt += 1
print(cnt)