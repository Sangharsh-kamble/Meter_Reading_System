import cv2
import numpy as np
import pytesseract
import os
from scipy import ndimage
from scipy.ndimage import interpolation as inter
from skimage.transform import radon
import warnings
kernel = np.ones((5,5),np.uint8)
warnings.filterwarnings("ignore", category=UserWarning)

def thresh(image, start = 15, end = 2):
    hsv = cv2.blur(image, (5, 5))
    h, s, v = cv2.split(hsv)
    thresh0 = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, start, end)
    thresh1 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, start, end)
    thresh2 = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, start, end)
    thresh12 = cv2.bitwise_or(thresh1, thresh2)
    thresh21 = cv2.bitwise_or(thresh2, thresh1)
    return thresh1

def rotation(image):
    if len(image.shape) < 3:
        img = image
        I = img
    elif len(image.shape) ==3:
        img = image
        I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = I.shape
    if (w > 640):
        I = cv2.resize(I, (640, int((h / w) * 640)))
    I = I - np.mean(I)
    sinogram = radon(I)
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    M = cv2.getRotationMatrix2D((w/2, h/2), 90 - rotation, 1)
    angle = 90 - rotation
    return angle

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    bestAngle = angles[scores.index(max(scores))]
    return bestAngle

def ocr(image_path):
    #img=image_path
    #img=cv2.resize(image_path, (150,50), interpolation = cv2.INTER_AREA)
    img =cv2.imread(image_path)
   # height, width, _ = img.shape[:3]
    img = cv2.resize(img, (250, 50))
    
    #print(img)
    rotate_angle=rotation(img)
    if rotate_angle==0:
        rotate_angle=correct_skew(img,delta=1, limit=5)
    start=rotate_angle-1
    end=rotate_angle+1
    print(rotate_angle) 
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #denoise gray
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, 10, 20)    
    #image erosion
    img_erosion = cv2.erode(denoised_gray, kernel, iterations=1)
    #image dilation
    img_dilation = cv2.dilate(img, kernel, iterations=1)  
    threshold_img=thresh(img_dilation,start = 11, end = 2)
    or_image = cv2.bitwise_not(img_erosion)
    #gray denoise image erusion
    gray_img_erosion = cv2.erode(denoised_gray, kernel, iterations=1)
    #gray denoise image dilation
    gray_img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)   
    image_list=(gray,img_dilation,gray_img_erosion,threshold_img,or_image,gray_img_dilation,or_image)
    for filtered in image_list:
            for i in range(-1,1,1):
                # print(i)
                height, width = filtered.shape[:2] # image shape has 3 dimensions
                image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
                rotation_mat = cv2.getRotationMatrix2D(image_center, i, 1)
           # rotation calculates the cos and sin, taking absolutes of those.
                abs_cos = abs(rotation_mat[0,0])
                abs_sin = abs(rotation_mat[0,1])
           # find the new width and height bounds
                bound_w = int(height * abs_sin + width * abs_cos)
                bound_h = int(height * abs_cos + width * abs_sin)
           # subtract old image center (bringing image back to origo) and adding the new image center coordinates
                rotation_mat[0, 2] += bound_w/2 - image_center[0]
                rotation_mat[1, 2] += bound_h/2 - image_center[1]
           # rotate image with the new bounds and translated rotation matrix
                rotated_mat = cv2.warpAffine(filtered, rotation_mat, (bound_w, bound_h))
                cv2.imshow("rotate image",rotated_mat)
                cv2.waitKey(0)
           # Paased filter image to ocr to get text from image
                tessdata_dir_config = "/usr/share/tesseract-ocr/4.00/tessdata/-l seg --oem 2 --psm 11"
                text = pytesseract.image_to_string(rotated_mat,lang= 'seg',config=tessdata_dir_config)
                print(text)
               
#print(ocr("/home/miscos/Downloads/3018.jpg"))

