
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pytesseract
import barcode_1 as br
import sevenSegmentOcr as socr

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
def play_video_with_cropping():
# Name of the directory containing the object detection module we're using
    MODEL_NAME = 'meter_models'
    VIDEO_NAME = '3.mp4'

# Grab path to current working directory
    CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'/home/miscos/Desktop/models/research/object_detection/meter_model/frozen_inference_graph.pb')

# Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'meter_data','/home/miscos/Desktop/models/research/object_detection/meter_data/object-detection.pbtxt')

# Path to video
    PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME,'/home/miscos/Downloads/cut_video.mp4')

# Number of classes the object detector can identify
    NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
    video = cv2.VideoCapture(PATH_TO_VIDEO)
    print("ok")
    i = 0
    while(video.isOpened()):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        if ret==True:

            frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})


        # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.70)
            height, width, channel = frame.shape
            ymin = int((boxes[0][0][0]*height))
            xmin = int((boxes[0][0][1]*width))
            ymax = int((boxes[0][0][2]*height))
            xmax = int((boxes[0][0][3]*width))
            Result = np.array(frame[ymin:ymax,xmin:xmax])
           # print(xmin,ymin,xmax,ymax)
        # All the results have been drawn on the frame, so it's time to display it.
            imS = cv2.resize(frame, (1200, 740))
            cv2.imshow('Object detector', imS)
            try:
                cv2.imshow("cropped",Result)
            except:
                pass
            labels = [category_index.get(i) for i in classes[0]]
            l=labels[0]
            l=l.get('name')
            #print(l)
            #print(scores[0])
           # print(boxes[0][0])
           # print(category_index)
            path = '/home/miscos/Desktop/models/research/object_detection/tester'
            path1 = '/home/miscos/Desktop/models/research/object_detection/screen_test'
            path2 = '/home/miscos/Desktop/models/research/object_detection/barcode_test'
  
            if scores[0][0] >= 0.8:
              if l=="screen":
                cv2.imwrite(os.path.join(path1,'img'+str(i)+'.jpg'),Result)
               #call the ocr function to read number from screen
               # reading=socr.ocr(Result)
              #  print("Reading",reading)
              elif l=="barcode":
                cv2.imwrite(os.path.join(path2,'img'+str(i)+'.jpg'),Result)
               #call the zbar library and barcode reader code to decode the barcode
               # try:
                 # serial_number=br.barcodeReader(Result)
                #  print("serial_number",serial_number)
                #except:
                #  pass
              else:
                pass
            if ret == False:
                break
            cv2.imwrite(os.path.join(path,'img'+str(i)+'.jpg'),Result)         
            i+=1

    #video.set(cv2.CAP_PROP_POS_MSEC,2000)
    #cv2.imwrite('img.png', Result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
    print("end1")
    video.release()
    cv2.destroyAllWindows()
    print("end2")

play_video_with_cropping()
