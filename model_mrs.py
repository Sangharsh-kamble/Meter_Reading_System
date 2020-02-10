import numpy as np
import pymysql
import sys
import shutil
import tensorflow as tf
import natsort
from PIL import Image
import cv2
import os
from os import *
import pytesseract
import glob
import time
#import pandas as pd
from functools import reduce
from custom_plate import url as ur
from custom_plate import do_image_conversion as dic
from custom_plate import allow_needed_values as anv
from custom_plate import test_db as db
from custom_plate import image_url as img
from custom_plate import remove_video as remo
from custom_plate import wait_for_video as wait
from custom_plate import frame_cutter as fps
from custom_plate import camera_link as camera
from custom_plate import release_camera as rel_cam
from custom_plate import camera_link as camera
from custom_plate import new_demo as nd
import threading
#from multiprocessing import Process
import datetime
import psutil
from custom_plate import repeat
import timeit




from utils import label_map_util
from utils import visualization_utils as vis_util

folder=camera.get_folder()
location_id=camera.get_cam_id()

print(str(folder))
#"""to store complete exceution flow to file"""
MODEL_NAME = "ssd_number_plate_graph"

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', '/home/miscos/Desktop/models/research/object_detection/custom_plate/Extra_Work/training/object-detection.pbtxt')
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

###organization_id=camera.get_cam_id()
rtsp_link_one=camera.get_camera_link()
#print(type(rtsp_link_one))

def load_image_into_numpy_array_updated(image):
  return np.array(image).astype(np.uint8)
#def load_image_into_numpy_array(image):
#  return np.array(image).astype(np.uint8)

def remove_images():
    dir_to_clean = str(folder)+"/allImages"

    l = os.listdir(dir_to_clean)

    l1=natsort.natsorted(os.listdir(dir_to_clean))


    for n in l1[0:10:1]:
        target = dir_to_clean + '/' + n
        if os.path.isfile(target):
            time.sleep(0.7)
            os.unlink(target)
        while len(l1)>=0:
            remove_images()


def compare(a, b):
    count = 0
    for x, y in zip(a, b):
        if x == y:
            count+=1
    return count
def data_compare(carReg,image_path):
    if len(carReg) < 8 and len(carReg) > 10:
        try:
            os.remove(image_path)
        except:
            pass

    elif len(carReg) <= 10 and len(carReg)>=8:
        try:
            shutil.move(image_path, str(folder)+"/crop_images")
        except shutil.Error as msg:
                pass

        db.psyco_insert_plate(carReg,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[2],location_id[1],"100","normal",camera.get_cam_id()[0])
            #method2()

def frame_cropping():

    print("frame cropping...")
    print("frame cropping link :",str(folder)+"/allImages")
    fps.frame_crop(rtsp_link_one,str(folder)+"/allImages")
    #rtsp://admin:admin@192.168.0.112:554/media/video1
def frame_cropping_one():

    print("frame cropping...")

    fps.frame_crop_one(rtsp_link_one,str(folder)+"/allImages")

#directory()
start=0
global PATH_TO_TEST_IMAGES_DIR
print(folder)
PATH_TO_TEST_IMAGES_DIR = str(folder)+"/allImages"
#end=(len([iq for iq in os.scandir(str(folder)+"/allImages")]))
outer_jump=10
pickup_frames=1

def method1():

    global outer_jump
    global start
    print("Welcome")
    print("outer_jump_value:",outer_jump)


    TEST_IMAGE_PATHS = np.array([os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(start,100000,outer_jump)])
    print(type(TEST_IMAGE_PATHS))
    IMAGE_SIZE = (8, 4)
    TEST_DHARUN=os.path.join('ssd_number_plate_graph')
    count = 0

    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        count=0
        d=0
        i=0
        c=0
        RTO_CODE = ["AP","AR","AS","BR","CG","GA","GJ","HR","HP","JK","JH","KA","KL","MP","MH",
                    "ML","MZ","NL","OD","PB","RJ","SK","TN","TR","UP","UK","UA","WB","TS","AN","CH",
                    "DN","DD","LD","DL","PY"]
        for c, image_path in np.ndenumerate(TEST_IMAGE_PATHS):
            global filename
            i += 1
            print(psutil.cpu_percent())
            #print(psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])
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
                image_np = load_image_into_numpy_array_updated(image)
            except UnboundLocalError as msg:
                pass
            image_np_expanded = np.expand_dims(image_np, axis=0)
            #output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            start_time = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            print('Iteration %d: %.3f sec'%(i, time.time()-start_time))
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
                filename = os.path.join(str(folder)+"crop_img_one/",'{}.jpg'.format(count))
                count += 1
                #filename = "/opt/lampp/htdocs/alpr/numberPlateDetection/crop_img_one/%d.jpg"%d
                cv2.imwrite(filename, img)
            #    d+=1
            except ValueError as msg:
                continue
            #start = time.process_time()
            carReg=nd.readNumbers(filename)
        #    print(carReg)

            print("number_plate:",carReg)
            if carReg[0:2] in RTO_CODE:
                try:
                    print("last number :",repeat.get_number()[-1])
                    print("match count :",compare(repeat.get_number()[-1],carReg))
                    if compare(repeat.get_number()[-1],carReg)!=0 and compare(repeat.get_number()[-1],carReg)<=7:
                        data_compare(carReg,image_path)
                    else:
                        pass
                except IndexError:
                    if carReg[0:2] in RTO_CODE:
                        db.psyco_insert_plate(carReg,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[2],location_id[1],"100","normal",camera.get_cam_id()[0])
                        try:
                            shutil.move(image_path, str(folder)+"/crop_images/")
                        except shutil.Error as msg:
                            pass
            else:
                continue

        #    print("output of file {} is : {}".format(filename,carReg))
            # try:
            #     if carReg in repeat.get_number():
            #         continue
            # except UnboundLocalError:
            #     pass



    try:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=18,
            min_score_thresh=0.5)
    except UnboundLocalError as msg:
            rel_cam.realese_camera_link(rtsp_link_one)
            print("outer completed")

def method2():

    global outer_jump
    global start
    print("Welcome")
    print("outer_jump_value:",outer_jump)


    TEST_IMAGE_PATHS = np.array([os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(start,100000,outer_jump)])
    print(type(TEST_IMAGE_PATHS))
    IMAGE_SIZE = (8, 4)
    TEST_DHARUN=os.path.join('ssd_number_plate_graph')
    count = 0

    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        count=0
        d=0
        i=0
        c=0
        for c, image_path in np.ndenumerate(TEST_IMAGE_PATHS):
            global filename
            i += 1
            print(psutil.cpu_percent())
            #print(psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])
            try:
                image = Image.open(image_path)
            except FileNotFoundError as msg:
                rel_cam.realese_camera_link(rtsp_link_one)
                break
            print("welcome")
            print(image_path)
            #time.sleep(1)
            try:
                image_np = load_image_into_numpy_array_updated(image)
            except UnboundLocalError as msg:
                pass
            image_np_expanded = np.expand_dims(image_np, axis=0)
            #output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            start_time = time.time()
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            #print(ok)
            print('Iteration %d: %.3f sec'%(i, time.time()-start_time))
        #    print([category_index.get(i) for i in classes[0]])
        #    print(scores)
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
                #print(img)
                filename = str(folder)+"/crop_img_one/%d.jpg"%d
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
            letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
            numbers = ["1","2","3","4","5","6","7","8","9","0"]
            RTO_CODE = ["AP","AR","AS","BR","CG","GA","GJ","HR","HP","JK","JH","KA","KL","MP","MH","MN",
                        "ML","MZ","NL","OD","PB","RJ","SK","TN","TR","UP","UK","UA","WB","TS","AN","CH",
                        "DN","DD","LD","DL","PY"]
            Non_first = ['E','F','I','U','V','X','Y','Z']
            Non_last  = ['C','E','F','I','M','O','Q','U','V','W','X']
            confusing_chars = {'2', 'Z', 'B', '8', 'D', '0', '5', 'S', 'Q', 'R', '7'}
            similar_characters = {'2':['Z'], 'Z':['2', '7'], '8':['B'], 'B':['8', 'R'], '5':['S'], 'S':['5'],
                         '0':['D', 'Q'], 'D':['0', 'Q'], 'Q':['D', '0'], '7':['Z']}

            if (len(carReg)>10 or len(carReg)<10):
                try:
                    os.remove(image_path)
                except:
                    pass


            elif len(carReg)==10:
                if carReg[2:4]=="Z3":
                    newline = carReg.replace('Z3','43')
                    print("Number_Plate :",newline)
                    base=os.path.basename(image_path)
                    start=(int(os.path.splitext(base)[0]))
                    print(start)
                    try:
                        shutil.move(image_path, str(folder)+'/crop_images')
                    except shutil.Error as msg:
                        pass
                    rep = newline
                    db.psyco_insert_plate(carReg,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[2],location_id[1],"100","normal",camera.get_cam_id()[0])
                    #method2()


                elif (carReg[0] in letters and carReg[1] in letters and carReg[2] in numbers and carReg[3] in numbers and carReg[4] in letters and carReg[5] in letters and          carReg[6] in numbers and carReg[7] in numbers and carReg[8] in numbers and          carReg[9] in numbers)==False:
                    try:
                        os.remove(image_path)
                    except:
                        pass

                elif len(carReg)==10:
      #print("Number_Plate :",carReg)
                    if (carReg[0] in letters and carReg[1] in letters and carReg[2] in numbers and carReg[3] in numbers and carReg[4] in letters and carReg[5] in letters and              carReg[6] in numbers and carReg[7] in numbers and carReg[8] in numbers and              carReg[9] in numbers)==True:
                        base=os.path.basename(image_path)
                        start=(int(os.path.splitext(base)[0]))
                        print(start)
                        try:
                            shutil.move(image_path, str(folder)+"/crop_images")
                        except shutil.Error as msg:
                            pass
                        output = carReg
                        print(type(output))

                        if output[0:2] in RTO_CODE:
                            print('Original Output : ',output)
                            new1 = output
                            db.psyco_insert_plate(carReg,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[2],location_id[1],"100","normal",camera.get_cam_id()[0])
                            #method2()
                        elif output[0:2] not in RTO_CODE:
                            repls = ('WH', 'MH'), ('HH', 'MH'),('LM','MH'),('MN','MH'),('MM','MH'),('MI','MH')
                            newline =  reduce(lambda a, kv: a.replace(*kv), repls, output)
                            new = newline
                            db.psyco_insert_plate(carReg,os.path.basename(image_path),os.path.basename(filename),datetime.datetime.now(),location_id[2],location_id[1],"100","normal",camera.get_cam_id()[0])

    try:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.90)
    except UnboundLocalError as msg:
            rel_cam.realese_camera_link(rtsp_link_one)
            print("outer completed")


if __name__ == '__main__':

  p1 = threading.Thread(target=frame_cropping,name='my_service')
  p2 = threading.Thread(target=method2,name='worker')
  p3 = threading.Thread(target=remove_images)



 # p4 = Process(target=method3)
  p1.start()
  #p2.start()
  time.sleep(2)
  p2.start()
  #time.sleep(30)
  #p3.start()
  p1.join()
  p2.join()
  #p3.join()
  #p4.join()


#method1()
#method2()
