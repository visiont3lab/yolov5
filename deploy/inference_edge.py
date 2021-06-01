#!/usr/bin/env python3
import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from openvino.inference_engine import IECore
#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append('/opt/intel/openvino/deployment_tools/inference_engine/demos/common/python')

import monitors
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

# Convert Pytorch model
# cd /opt/intel/openvino_2021.3.394/deployment_tools/tools/model_downloader
# python3 converter.py --name  best_model.pt  --download_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models  --output_dir /opt/intel/openvino_2021.3.394/ws/sample-app/models
# 
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model.html

"""
# Conversion
cd /opt/intel/openvino/deployment_tools/model_optimizer
python3 mo.py --input_model /ws/models/best.onnx --output_dir /ws/models
cd /opt/intel/openvino_2021.3.394/ws
python3 inference_edge.py \
    -i 0  \
    -m /opt/intel/openvino_2021.3.394/ws/models/FP16/best.xml  \
    -d MYRIAD 
"""
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


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to a test image file.",
                      required=True, type=str)
    args.add_argument("-m", "--model",
                      help="Required. Path to an .xml file with a pnet model.",
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str, metavar='"<device>"')
    args.add_argument('--loop', default=False, action='store_true',
                       help='Optional. Enable reading the input in a loop.')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser

def main():
    metrics = PerformanceMetrics()
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")

    # IE Core
    ie = IECore()

    # Read IR
    log.info("Loading network files:\n\t{}".format(args.model))
    net = ie.read_network(args.model)

    # Input Blobs
    log.info("Preparing input blobs")
    net_input_blob = next(iter(net.input_info))

    # Output Blobs
    log.info("Preparing output blobs")
    for name, blob in net.outputs.items():
        print(f"--> Blob shape: {blob.shape}")
        print(f"--> Output layer: {name}")

    # Image Capture (RTSP-Video-Image)
    cap = open_images_capture(args.input, args.loop)
    next_frame_id = 0
    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    presenter = None
    video_writer = cv2.VideoWriter()


    # Load network
    log.info("Loading Net model to the plugin")
    w, h = (640,640)

    net.reshape({net_input_blob: [1,3, w, h]})  # Change weidth and height of input blob
    exec_net = ie.load_network(network=net, device_name=args.device)

    while True:
        start_time = perf_counter()
        origin_image = cap.read()
        if origin_image is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            break
        if next_frame_id == 0:
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(origin_image.shape[1] / 4), round(origin_image.shape[0] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                    cap.fps(), (origin_image.shape[1], origin_image.shape[0])):
                raise RuntimeError("Can't open video writer")
        next_frame_id += 1

        rgb_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        oh, ow, _ = rgb_image.shape

        # ----------------------------------------------------------------
        # Net stage
        # ----------------------------------------------------------------
        t0 = cv2.getTickCount()
        image = cv2.resize(rgb_image, (w, h))
        image = image / 255
        image = image.transpose((2, 0, 1))
        net_input = np.expand_dims(image, axis=0)
        net_res = exec_net.infer(inputs={net_input_blob: net_input})["688"]

        pred_onx = net_res[0]                      # x(1, 25200, 7) to x(25200, 7)
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
        
            if ((scores[i] > 0.1) and (scores[i] <= 1.0)):
                
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
                            
                cv2.rectangle(origin_image, (xmin,ymin), (xmax,ymax), class_colors[classes[i]], 2)
        # ----------------------------------------------------------------

        infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()  # Record infer time
        cv2.putText(origin_image, 'summary: {:.1f} FPS'.format(1.0 / infer_time),(5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit - 1):
            video_writer.write(origin_image)

        if not args.no_show:
            cv2.imshow('Net Results', origin_image)
            key = cv2.waitKey(1)
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

        metrics.update(start_time, origin_image)

    metrics.print_total()


if __name__ == '__main__':
    sys.exit(main() or 0)
