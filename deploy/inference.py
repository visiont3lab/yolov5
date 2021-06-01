import numpy
import onnxruntime as rt
import cv2
import os
import numpy as np
import torch, torchvision
import time

# Non Max suppression
# https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/

# https://stackoverflow.com/questions/65824714/process-output-data-from-yolov5-tflite

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

class_names = ['dado-M3', 'dado-M5', 'dado-M6', 'dado-M8', 'rondella-M3', 'rondella-M6', 'vite-croce', 'vite-esagonale', 'vite-taglio-grande', 'vite-taglio-piccola']
class_colors =  [(125,0,0), (125,0,50), (125,100,0), (125,0,125),(0,255,0), (255,255,0), (125,0,255), (125,255,255), (0,0,255), (125,0,65)]

sess = rt.InferenceSession("../runs/train/sapera_yolov5m_640/weights/best.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
input_shapes = sess.get_inputs()[0].shape
output_shapes = sess.get_outputs()[0].shape
c, h ,w = (input_shapes[1],input_shapes[2],input_shapes[3])

print(f"input name: {input_name}")
print(f"input shapes: {input_shapes}")

print(f"output name: {label_name}")
print(f"output shapes: {output_shapes}")

cv2.namedWindow('Image',cv2.WINDOW_NORMAL) 
folder_path = "../datasets/dadi/dataset_dadi_sapera"
names = os.listdir(folder_path)
for name in names:
    filename = os.path.join(folder_path,name)        
    print(filename)
    frame = cv2.imread(filename, cv2.IMREAD_COLOR)
    iH, iW = (frame.shape[0],frame.shape[1])

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(h,w))
    im = im / 255.0  # 0 - 255 to 0.0 - 1.0   
    im = np.transpose(im, (2, 0,1))
    im = im[np.newaxis,:,:,:]
    pred_onx = sess.run( [label_name], {input_name: im.astype(np.float32)})[0]
    
    pred_onx = pred_onx[0]                      # x(1, 25200, 7) to x(25200, 7)
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
     
        if ((scores[i] > 0.9) and (scores[i] <= 1.0)):
            
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
                        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), class_colors[classes[i]], 2)
    
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
            
