import os
import cv2
from tqdm import tqdm
import shutil
import traceback
import re
import random

import cv2
import mediapipe as mp

def scale_bbox(bbox, scale_factor=1.2):
    x, y, w, h = bbox
    new_w = w * scale_factor
    new_h = h * scale_factor
    new_x = x - (new_w - w) / 2
    new_y = y - (new_h - h) / 2
    return int(new_x), int(new_y), int(new_w), int(new_h)

def detect_faces(image_path1, image_path2, save_path1, save_path2):
    # Khởi tạo đối tượng phát hiện khuôn mặt từ MediaPipe
    face_detection = mp.solutions.face_detection.FaceDetection()

    # Đọc hình ảnh
    image = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Chuyển đổi hình ảnh sang RGB (MediaPipe yêu cầu định dạng này)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt trong hình ảnh
    results = face_detection.process(image_rgb)

    # Lấy ra các khuôn mặt đã phát hiện được cùng với hình ảnh của chúng
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            # Cắt ảnh của khuôn mặt từ hình ảnh gốc
            x, y, w, h = scale_bbox(bbox)
            face_image1 = image[y:y+h, x:x+w]
            face_image2 = image2[y:y+h, x:x+w]
            break
    cv2.imwrite(save_path1, face_image1)
    cv2.imwrite(save_path2, face_image2)

# path_folder = r'last_data'
# path_folder_cropped = r'last_data_cropped'
# list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
# cnt = 0
# for id in (list_folder_id):
#     try:
#         path_folder_id = os.path.join(path_folder, id)
#         list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]
#         if(len(list_folder_id_time) > 1):
#             for time_id in list_folder_id_time:
#                 path_folder_id_time = os.path.join(path_folder_id, time_id)
                
#     except:
#         print(id)
#         traceback.print_exc()

# print(cnt)

path_folder = 'last_data'
folder_new_3d = 'data_train_3d'
folder_new_2d = 'data_train_2d'

list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]
for id in tqdm(list_folder_id):
    try:
        cnt = 0
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]
        folder_new_id_2d = os.path.join(folder_new_2d, id)
        folder_new_id_3d = os.path.join(folder_new_3d, id)
        if len(list_folder_id_time) > 1:
            if len(list_folder_id_time) > 5:
                list_folder_id_time = random.sample(list_folder_id_time, 5)
        else:
            continue
        os.makedirs(folder_new_id_2d, exist_ok=True)
        os.makedirs(folder_new_id_3d, exist_ok=True)
        for time_id in list_folder_id_time:
            cnt+=1
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            image_path1 = os.path.join(path_folder_id_time, 'merge.png')
            image_path2 = os.path.join(path_folder_id_time, 'result.png')
            save_path1 = os.path.join(folder_new_id_2d, f'{cnt}.png')
            save_path2 = os.path.join(folder_new_id_3d, f'{cnt}.png')
            detect_faces(image_path1, image_path2, save_path1, save_path2)
    except:
        print(id)
        traceback.print_exc()