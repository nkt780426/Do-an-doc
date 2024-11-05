from photostereo import photometry
import cv2 as cv
import time
import numpy as np
import os
from tqdm import tqdm
import traceback

IMAGES = 4

def run_script(root_fold):
    light_manual = False

    #Load input image array
    image_array = []

    list_name_image = [x for x in os.listdir(root_fold) if x.endswith('.bmp') or x.endswith(".png")]
    image_array = [(cv.imread(os.path.join(root_fold, x), cv.IMREAD_GRAYSCALE)) for x in list_name_image]

    # list_name_image = [x for x in os.listdir(os.path.join('Photoface_dist\PhotofaceDB',root_fold[11:])) if x.endswith('.bmp') or x.endswith(".png")]
    # image_array = [(cv.imread(os.path.join(os.path.join('Photoface_dist\PhotofaceDB',root_fold[11:]), x), cv.IMREAD_GRAYSCALE)) for x in list_name_image]

    myps = photometry(IMAGES, False)

    if light_manual:
        # SETTING LIGHTS MANUALLY
        # slants = [71.4281, 66.8673, 67.3586, 67.7405]
        # tilts = [140.847, 47.2986, -42.1108, -132.558]

    #     LightAngle = [
    #    90-15.7 360-117.3;
    #    90-17.9 360-50.3;
    #    90-18.4 360-302.8;
    #    90-20.0 360-229.3;
    #    ]; 

        # LightAngle = [
        #     90-36 360-134;
        #     90-42 360-47;
        #     90-40 360-313;
        #     90-36 360-233];

        # slants = [90-36, 90-42, 90-40, 90-36]
        # tilts = [180-134, 180-47, 180-313, 180-233]
        
        slants = [90-36, 90-42, 90-40, 90-36]
        tilts = [180-134, 180-47, 180-313, 180-233]

        # slants = [36, 42, 40, 36]
        # tilts = [180-134, 180-47, 180-313, 180-233]

        slants = [36, 42, 40, 36]
        tilts = [90-134, 90-47, 90-313, 90-233]

        # slants = [15.7, 17.9, 18.4, 20]
        # tilts = [360-117.3, 360-50.3, 360-302.8, 360-229.3]

        myps.setlmfromts(tilts, slants)
        print(myps.settsfromlm())
    else:
        # # LOADING LIGHTS FROM FILE
        # fs = cv.FileStorage(root_fold + "LightMatrix.yml", cv.FILE_STORAGE_READ)
        # fn = fs.getNode("Lights")
        # light_mat = fn.mat()
        # myps.setlightmat(light_mat)
        # #print(myps.settsfromlm())
        
        # Load light_source.txt
        with open(os.path.join(root_fold, 'LightSource.txt'), 'r') as f:
            data_light = f.read()
            data_light = [float(x) for x in data_light.split(' ')]
            slants = [data_light[0], data_light[2], data_light[4], data_light[6]]
            tilts = [90-data_light[1], 90-data_light[3], 90-data_light[5], 90-data_light[7]]
            # print(slants, tilts)
            myps.setlmfromts(tilts, slants)


    mask = np.ones_like(image_array[0]) * 255
    normal_map = myps.runphotometry(image_array, np.asarray(mask, dtype=np.uint8))
    normal_map = cv.normalize(normal_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    # albedo = myps.getalbedo()
    # albedo = cv.normalize(albedo, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    # gauss = myps.computegaussian()
    # med = myps.computemedian()
    cv.imwrite(os.path.join(root_fold, 'result.png'), normal_map)    # cv.imshow("normal", normal_map)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

def run_script1(root_fold):
    light_manual = False

    #Load input image array
    image_array = []

    # list_name_image = [x for x in os.listdir(root_fold) if x.endswith('.bmp') or x.endswith(".png")]
    # image_array = [(cv.imread(os.path.join(root_fold, x), cv.IMREAD_GRAYSCALE)) for x in list_name_image]

    list_name_image = [x for x in os.listdir(os.path.join('Photoface_dist\PhotofaceDB',root_fold[11:])) if x.endswith('.bmp') or x.endswith(".png")]
    image_array = [(cv.imread(os.path.join(os.path.join('Photoface_dist\PhotofaceDB',root_fold[11:]), x), cv.IMREAD_GRAYSCALE)) for x in list_name_image]

    myps = photometry(IMAGES, False)

    if light_manual:
        # SETTING LIGHTS MANUALLY
        # slants = [71.4281, 66.8673, 67.3586, 67.7405]
        # tilts = [140.847, 47.2986, -42.1108, -132.558]

    #     LightAngle = [
    #    90-15.7 360-117.3;
    #    90-17.9 360-50.3;
    #    90-18.4 360-302.8;
    #    90-20.0 360-229.3;
    #    ]; 

        # LightAngle = [
        #     90-36 360-134;
        #     90-42 360-47;
        #     90-40 360-313;
        #     90-36 360-233];

        # slants = [90-36, 90-42, 90-40, 90-36]
        # tilts = [180-134, 180-47, 180-313, 180-233]
        
        slants = [90-36, 90-42, 90-40, 90-36]
        tilts = [180-134, 180-47, 180-313, 180-233]

        # slants = [36, 42, 40, 36]
        # tilts = [180-134, 180-47, 180-313, 180-233]

        slants = [36, 42, 40, 36]
        tilts = [90-134, 90-47, 90-313, 90-233]

        # slants = [15.7, 17.9, 18.4, 20]
        # tilts = [360-117.3, 360-50.3, 360-302.8, 360-229.3]

        myps.setlmfromts(tilts, slants)
        print(myps.settsfromlm())
    else:
        # # LOADING LIGHTS FROM FILE
        # fs = cv.FileStorage(root_fold + "LightMatrix.yml", cv.FILE_STORAGE_READ)
        # fn = fs.getNode("Lights")
        # light_mat = fn.mat()
        # myps.setlightmat(light_mat)
        # #print(myps.settsfromlm())
        
        # Load light_source.txt
        with open(os.path.join(root_fold, 'LightSource.txt'), 'r') as f:
            data_light = f.read()
            data_light = [float(x) for x in data_light.split(' ')]
            slants = [data_light[0], data_light[2], data_light[4], data_light[6]]
            tilts = [90-data_light[1], 90-data_light[3], 90-data_light[5], 90-data_light[7]]
            print(slants, tilts)
            myps.setlmfromts(tilts, slants)


    mask = np.ones_like(image_array[0]) * 255
    normal_map = myps.runphotometry(image_array, np.asarray(mask, dtype=np.uint8))
    normal_map = cv.normalize(normal_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    # albedo = myps.getalbedo()
    # albedo = cv.normalize(albedo, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    # gauss = myps.computegaussian()
    # med = myps.computemedian()
    cv.imwrite(os.path.join(root_fold, 'result.png'), normal_map)
    # cv.imshow("normal", normal_map)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

path_folder = r'final_data'
list_folder_id = [x for x in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, x))]

for id in tqdm(list_folder_id):
    try:
        path_folder_id = os.path.join(path_folder, id)
        list_folder_id_time = [x for x in os.listdir(path_folder_id) if os.path.isdir(os.path.join(path_folder_id ,x))]

        for time_id in list_folder_id_time:
            path_folder_id_time = os.path.join(path_folder_id, time_id)
            run_script(path_folder_id_time)
    except:
        print(id)
        traceback.print_exc()

# run_script(r"final_data\1043\2008-02-18_15-21-00")
# run_script1(r"final_data\1043\2008-02-18_15-21-00")

