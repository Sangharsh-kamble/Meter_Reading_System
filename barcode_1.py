from pyzbar.pyzbar import decode
import cv2
import numpy as np
import glob


def barcodeReader(img):
    #image=img
    image=cv2.imread(img)
    bgr = (8, 70, 208)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray_img)

    for decodedObject in barcodes:
        points = decodedObject.polygon

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, (0, 255, 0), 3)

    for bc in barcodes:
       # cv2.putText(frame, bc.data.decode("utf-8") + " - " + bc.type, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   # bgr, 2)

        return "Barcode: {}".format(bc.data.decode("utf-8"))

#for img in glob.glob("/home/miscos/Desktop/models/research/object_detection/barcode_test/*.jpg"):
  #  n= barcodeReader(img)
  #  print(n)

#print(barcodeReader("/home/miscos/Desktop/models/research/object_detection/barcode_test/img1200.jpg"))
