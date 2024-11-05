import os
import cv2
from tqdm import tqdm
import shutil
import traceback

def merge_image(list_image):
    image1, image2, image3, image4 = tuple(list_image)
    if image1.shape == image2.shape == image3.shape == image4.shape:
        blended_image = cv2.addWeighted(image1, 0.25, image2, 0.25, 0)
        blended_image = cv2.addWeighted(blended_image, 0.5, image3, 0.25, 0)
        blended_image = cv2.addWeighted(blended_image, 0.5, image4, 0.25, 0)
        return cv2.cvtColor(blended_image, cv2.COLOR_BGR2GRAY)

path_folder = r'final_data'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
data_ok = []
cnt = 0
for id in tqdm(list_folder_id):
    try:
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            check_ok = True
            with open(os.path.join(path_folder_id_time, 'LightSource.txt'), 'r') as f:
                data_light = f.read()
                data_light = [float(x) for x in data_light.split(' ')]
                for x in data_light[::2]:
                    if x < 20:
                        check_ok = False
                        print(data_light)
                        break
            if check_ok:
                folder_new = os.path.join("last_data" + path_folder_id_time[10:])
                os.makedirs(folder_new, exist_ok=True)
                list_name_image = [x for x in os.listdir(path_folder_id_time) if x.startswith("im")]
                list_image = [cv2.imread(os.path.join(path_folder_id_time, x)) for x in list_name_image]
                merge_result = merge_image(list_image)
                cv2.imwrite(os.path.join(folder_new, 'merge.png'), merge_result)
                shutil.copy(os.path.join(path_folder_id_time, 'result.png'), os.path.join(folder_new, 'result.png'))
    except:
        print(id)
        traceback.print_exc()
