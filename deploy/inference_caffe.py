import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# OpenCv inference engine
# https://stackoverflow.com/questions/57007007/openvino-how-to-build-opencv-with-inference-engine-to-enable-loading-models-fro
# pip3 uninstall opencv-python
# pip3 uninstall opencv-contrib-python
# pip3 install opencv-python-inference-engine
# pip install cv2-plt-imshow
 
# https://www.pyimagesearch.com/2020/01/06/raspberry-pi-and-movidius-ncs-face-recognition/

 # Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[0,:]
    y1 = boxes[1,:]
    x2 = boxes[2,:]
    y2 = boxes[3,:]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[:,pick].astype("int"), pick


prototxt="./models/FP16/best.xml"
model="./models/FP16/best.bin"
 
class_names = ['dado-M3', 'dado-M5', 'dado-M6', 'dado-M8', 'rondella-M3', 'rondella-M6', 'vite-croce', 'vite-esagonale', 'vite-taglio-grande', 'vite-taglio-piccola']
class_colors =  [(125,0,0), (125,0,50), (125,100,0), (125,0,125),(0,255,0), (255,255,0), (125,0,255), (125,255,255), (0,0,255), (125,0,65)]
conf = 0.8
 
net = cv2.dnn.readNet(prototxt, model)

# https://medium.com/sclable/intel-openvino-with-opencv-f5ad03363a38
# DNN_TARGET_CPU, DNN_TARGET_OPENCL (for Intel GPU FP32), DNN_TARGET_OPENCL_FP16 (for Intel GPU FP16, preferred), DNN_TARGET_MYRIAD (for NCS 2), DNN_TARGET_FPGA, DNN_TARGET_CUDA (for NVIDIA GPU FP 32) and DNN_TARGET_CUDA_FP16 (for NVIDIA GPU FP16, preferred).
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD) # NCS 2
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_VPU)
 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    iH, iW = (frame.shape[0],frame.shape[1])
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(640,640))

    #frame = imutils.resize(frame, width=400)
    t1 = cv2.getTickCount()

    (h, w) = im.shape[:2]
    blob = cv2.dnn.blobFromImage(im,1/255,size=(im.shape[1],im.shape[0]))

    net.setInput(blob)
    detections = net.forward()

    pred_onx = detections[0]                      # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(pred_onx[:, :4])       # boxes  [25200, 4]
    scores = np.squeeze( pred_onx[:, 4:5])    # confidences  [25200, 1]
    
    classdata = pred_onx[:, 5:]               # get classes
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    classes = np.array(classes)
    
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    xx, yy, ww, hh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] #xywh
    xyxy = np.array([xx - ww / 2, yy - hh / 2, xx + ww / 2, yy + hh / 2])  # xywh to xyxy   [4, 25200]  # coordinate with respect to image model (h,w)

    # --- Data after nms
    pick = [idx for idx in range(0,len(scores)) if scores[idx] > 0.9]
    xyxy = xyxy[:,pick] 
    scores = scores[pick]
    classes = classes[pick]
    
    # -- NMS
    xyxy, pick = non_max_suppression_fast(xyxy, overlapThresh=0.5)
    scores = scores[pick]
    classes = classes[pick]

    for i in range(len(scores)):
     
        if ((scores[i] > conf) and (scores[i] <= 1.0)):
            
            # Normalize in range 0-1
            xmin = (xyxy[0][i] / h )
            ymin = (xyxy[1][i] / w )
            xmax = (xyxy[2][i] / h )
            ymax = (xyxy[3][i] / w )

            # Bring 0-1 coordinate in image cooridnates
            xmin = int(max(1,xmin*iW))
            ymin = int(max(1,ymin*iH))
            xmax = int(min(iW,xmax*iW))
            ymax = int(min(iH,ymax*iH))

            idx = classes[i]
            print(class_names[idx],np.round(scores[i],2),[xmin,ymin,xmax,ymax])
                        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), class_colors[idx], 2)            
            cv2.putText(frame, class_names[idx], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[idx], 2)
            #cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

 
    cv2.imshow("Frame", frame)
    cv2.waitKey(33)
    #plt.imshow(frame)
    #plt.pause(0.001)