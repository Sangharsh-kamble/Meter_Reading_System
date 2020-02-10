import numpy as np
import os
import pymysql
import sys
import shutil
import tensorflow as tf
import natsort
from PIL import Image
import cv2
from os import *
import pytesseract
import glob
import time
import pandas as pd
from functools import reduce
from custom_plate import url as ur

#from custom_plate import do_image_conversion as dic
from custom_plate import allow_needed_values as anv
from custom_plate import test_db as db
from custom_plate import image_url as img
from custom_plate import remove_video as remo
from custom_plate import wait_for_video as wait
from custom_plate import frame_cutter as fps
from custom_plate import camera_link as camera
from custom_plate import repeat
from custom_plate import release_camera as rel_cam
from scipy import ndimage
#from custom_plate import night_mode as nm
from multiprocessing import Process
import gc
import datetime
import logging
import psutil
from google.cloud import vision
from google.cloud.vision import types
import io

sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util
gc.collect()



#"""to store complete exceution flow to file"""
logging.basicConfig(filename='alpr.txt' ,level=logging.DEBUG ,format="%(asctime)s:%(levelname)s:%(message)s")
logging.info("New request came")
MODEL_NAME = 'ssd_number_plate_graph'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', '/home/miscos/Desktop/models/research/object_detection/custom_plate/Extra_Work/training/object-detection.pbtxt')
NUM_CLASSES = 1



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

rtsp_link_one=camera.get_camera_link()

def load_image_into_numpy_array(image):
  return np.array(image).astype(np.uint8)

def remove_images():
    dir_to_clean = '/opt/lampp/htdocs/alpr/numberPlateDetection/allImages'
    l = os.listdir(dir_to_clean)
    l1=natsort.natsorted(os.listdir(dir_to_clean))

    for n in l1[0:10:1]:
        target = dir_to_clean + '/' + n
        if os.path.isfile(target):
            time.sleep(0.4)
            os.unlink(target)
        while len(l1)>=0:
            remove_images()

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num
def detect_text(path,count):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
#    print('Texts:')

    for text in texts:
#        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

#        print('bounds: {}'.format(','.join(vertices)))
    count=count+1
    return text.description
#detect_text("/home/miscos/Desktop/1.jpeg")
def ocr(img,count):
    path=img
    letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    numbers = ["1","2","3","4","5","6","7","8","9","0"]
    kernel = np.ones((5,5),np.uint8)

    result=[]
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, 10, 20)
    denoise_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,15)
    img_erosion = cv2.erode(denoise_img, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    gray_img_erosion = cv2.erode(denoised_gray, kernel, iterations=1)
    gray_img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    dst = cv2.GaussianBlur(denoised_gray,(3,3),cv2.BORDER_DEFAULT)
    bilFilter = cv2.bilateralFilter(denoised_gray,9,75,75)
    image_list=(denoised_gray,img_dilation,gray_img_dilation,dst,bilFilter)

    for filter in image_list:
        for i in range(-7,-3,1):
           rotated = ndimage.rotate(filter, i)
           tessdata_dir_config = "/usr/share/tesseract-ocr/4.00/tessdata/-l eng --oem 1 --psm 11"
           try:
               text = pytesseract.image_to_string(rotated,lang= 'eng',config=tessdata_dir_config)
           except pytesseract.pytesseract.TesseractNotFoundError as msg:
               pass
           carReg=anv.catch_rectify_plate_characters(text)
           try:
               if len(carReg)==10 or len(carReg)==9 or len(carReg)==8:
                   if (carReg[0] in letters and carReg[1] in letters and carReg[2] in numbers and carReg[3] in numbers and carReg[4] in letters and carReg[5] in letters and carReg[6] in numbers and carReg[7] in numbers and carReg[8] in numbers and carReg[9] in numbers)==True:
                       result.append(carReg)
                       break
                   elif (carReg[0] in letters and carReg[1] in letters and carReg[2] in numbers and carReg[3] in numbers and carReg[4] in letters and carReg[5] in letters and carReg[6] in numbers and carReg[7] in numbers and carReg[8] in numbers and carReg[9] in numbers)==False:
                       break
                   else:
                       break
           except IndexError:
                pass

        if len(result)!=0:
             break

    #print(set(result))
    count=count+1
    return most_frequent(result) if len(result)>0 else ''

def frame_cropping():
    print("frame cropping...")
    fps.frame_crop(rtsp_link_one,'/opt/lampp/htdocs/alpr/numberPlateDetection/allImages')
    #rtsp://admin:admin@192.168.0.112:554/media/video1
def frame_cropping_one():
    print("frame cropping...")
    fps.frame_crop_one('/home/miscos/Desktop/models/research/object_detection/alpr1.mp4','/opt/lampp/htdocs/numberPlateDetection/allImages_one')

location_id=camera.get_cam_id()


global PATH_TO_TEST_IMAGES_DIR
PATH_TO_TEST_IMAGES_DIR = '/opt/lampp/htdocs/alpr/numberPlateDetection/allImages'
end=(len([iq for iq in os.scandir('/opt/lampp/htdocs/alpr/numberPlateDetection/allImages')]))
outer_jump=10
pickup_frames=1
start=0
#remo.video_remove("/opt/lampp/htdocs/numberPlateDetection/allVideos")
def method1():

    global outer_jump
    global start
    print("Welcome")
    print("outer_jump_value:",outer_jump)


    TEST_IMAGE_PATHS = (os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(start,100000,outer_jump))
    IMAGE_SIZE = (8, 4)
    TEST_DHARUN=os.path.join('ssd_number_plate_graph')
    count = 0


    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            count=0
            d=0
            for image_path in TEST_IMAGE_PATHS:
                global filename
                print(psutil.cpu_percent())
                #print(psutil.virtual_memory())  # physical memory usage
                print('memory % used:', psutil.virtual_memory()[2])
                logging.info("Inter loop started")
                try:
                    print()
                    image = Image.open(image_path)
                except FileNotFoundError as msg:
                    rel_cam.realese_camera_link(rtsp_link_one)
                    break
                print("welcome")
                print(image_path)
                #time.sleep(1)
                try:
                    image_np = load_image_into_numpy_array(image)
                except UnboundLocalError as msg:
                    pass
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    (detection_boxes, detection_scores, detection_classes, num_detections),
                    feed_dict={image_tensor: image_np_expanded})
                ymin = boxes[0,0,0]
                xmin = boxes[0,0,1]
                ymax = boxes[0,0,2]
                xmax = boxes[0,0,3]
                try:
                    (im_width, im_height) = image.size
                    (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                    cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
                    img_data = sess.run(cropped_image)
                    img = img_data
                    filename = "/opt/lampp/htdocs/alpr/numberPlateDetection/crop_img_one/%d.jpg"%d
                    cv2.imwrite(filename, img)
                    d+=1

                except ValueError as msg:
                    continue

                count = 0
                filename = dic.yo_make_the_conversion(img_data, count)
                print(filename)
                            #print("file_name_:",os.path.basename(filename))
                tessdata_dir_config = "/usr/share/tesseract-ocr/4.00/tessdata/-l eng --psm 10"
                try:
                    text = pytesseract.image_to_string(Image.open(filename),lang= 'eng',config=tessdata_dir_config)
                except pytesseract.pytesseract.TesseractNotFoundError as msg:
                    continue
                count+=1
                carReg = anv.catch_rectify_plate_characters(text)
                print(carReg)
                    try:
                        if carReg in repeat.get_number():
                            continue
                        except UnboundLocalError:
                            pass
                db.psyco_insert_plate(rep,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[1],location_id[0],"100","normal")

        try:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=18,
                min_score_thresh=0.90)
        except UnboundLocalError as msg:
                rel_cam.realese_camera_link(rtsp_link_one)
                #logging.exception(msg)
                print("outer completed")
    output_dir = "/home/miscos/Desktop/models/research/object_detection/filter_folder"



if __name__ == '__main__':
  p1 = Process(target=frame_cropping)
  p2 = Process(target=method1)
  p3 = Process(target=remove_images)

  p1.start()
  #p2.start()
  time.sleep(2)
  p2.start()
  time.sleep(30)
  p3.start()
  p1.join()
  p2.join()
  p3.join()
  #p4.join()
