import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import traceback

range_shift = 20

# Gamma trong xử lý ảnh là thông số điều chỉnh độ sáng và độ tương phản của ảnh. Nó là một yếu tố quan trọng trong quản lý ánh sáng trong ảnh kỹ thuật số và có ảnh hưởng lớn đến cách mà ảnh được hiển thị trên màn hình.
# Gamma là một hàm toán học mô tả mối quan hệ giữa độ sáng của tín hiệu đầu vào và độ sáng đầu ra mà mắt người thấy. Cụ thể, nó thể hiện độ phi tuyến giữa các giá trị pixel (tín hiệu đầu vào) và độ sáng (tín hiệu đầu ra).
# Nếu giá trị gamma > 1: Tăng cường độ sáng của các điểm sáng và làm cho các vùng tối trở nên tối hơn.
# Nếu giá trị gamma < 1: Giảm độ sáng của các điểm sáng và làm cho các vùng tối trở nên sáng hơn.
def pre_process(image):
    equalized_image = adjust_gamma(image, 1.2)
    return equalized_image

def histogram_equalization(image):
    # Chuyển ảnh về ảnh xám nếu ảnh màu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Cân bằng histogram
    equalized_image = cv2.equalizeHist(image)
    
    return equalized_image

def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def count_common_edges(edges1, edges2):
    common_edges = cv2.bitwise_and(edges1, edges2)
    contours, _ = cv2.findContours(common_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_common_edges = 0
    for contour in contours:
        if len(contour) >= 5:
            num_common_edges += len(contour)
    return num_common_edges

scores = np.zeros((range_shift*2, range_shift*2))  # Tạo một ma trận để lưu kết quả

def cal_crop_shift(image1, image2):
    # image1 = histogram_equalization(image1)
    # image2 = histogram_equalization(image2)
    threshold1 = 50
    threshold2 = 250
    gamma = 1
    image1 = adjust_gamma(image1, 1)
    image2 = adjust_gamma(image2, 1)    
    edges1 = cv2.Canny(image1, threshold1, threshold2)
    edges2 = cv2.Canny(image2, threshold1, threshold2)

    crop_edges1 = edges1[range_shift:-range_shift, range_shift:-range_shift]
    max_score = 0
    best_x_shift, best_y_shift = 0,0
    for x_shift in range(-range_shift, range_shift):
        for y_shift in range(-range_shift, range_shift):
            edges_cal = edges2[range_shift+x_shift:-range_shift+x_shift, range_shift+y_shift:-range_shift+y_shift]
            # common_edges = cv2.bitwise_and(crop_edges1, edges_cal)
            # score = cv2.countNonZero(common_edges)
            score = count_common_edges(crop_edges1, edges_cal)
            scores[x_shift + range_shift][y_shift + range_shift] = score
            if score > max_score:
                best_x_shift, best_y_shift = x_shift, y_shift
                max_score = score
    im1_crop = image1[range_shift:-range_shift, range_shift:-range_shift, :]
    im2_crop = image2[range_shift+best_x_shift:-range_shift+best_x_shift, range_shift+best_y_shift:-range_shift+best_y_shift, :]
    return max_score, best_x_shift, best_y_shift, im1_crop, im2_crop

path_folder = r'C:\Users\LAMHN\Documents\DoAn_KiSu\Photoface_dist\PhotofaceDB'
crop_path_folder = r'C:\Users\LAMHN\Documents\DoAn_KiSu\cropped_data'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]

for id in tqdm(list_folder_id):
    try:
        # vao trong id
        # tao folder id crop
        crop_path_folder_id = os.path.join(crop_path_folder, id)
        if os.path.exists(crop_path_folder_id):
            shutil.rmtree(crop_path_folder_id)
        os.mkdir(crop_path_folder_id)
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            crop_path_folder_id_time = os.path.join(crop_path_folder_id, time_id)
            if os.path.exists(crop_path_folder_id_time):
                shutil.rmtree(crop_path_folder_id_time)
            os.mkdir(crop_path_folder_id_time)
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            list_name_image = [x for x in os.listdir(path_folder_id_time) if x.endswith(".bmp")]

            list_image = [pre_process(cv2.imread(os.path.join(path_folder_id_time, x))) for x in list_name_image]
            result_crop = []
            for i in range(1, len(list_image)):
                max_score, best_x_shift, best_y_shift, re1, re2 = cal_crop_shift(list_image[0], list_image[i])
                list_name_image[i] = f'im{i}_{best_x_shift}_{best_y_shift}_{max_score}.png'
                if i == 1:
                    list_name_image[0] = f'im0.png'
                    result_crop.append(re1)
                result_crop.append(re2)
            for indexx in range(len(result_crop)):
                cv2.imwrite(os.path.join(crop_path_folder_id_time, list_name_image[indexx]), result_crop[indexx])
    except:
        print(id)
        traceback.print_exc()