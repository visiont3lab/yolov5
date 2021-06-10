# Custom Object detection Using yolov5


> Inference custom yolov5 using fastapi and ray

[Yolov5 Roboflow tutorial](https://models.roboflow.com/object-detection/yolov5)

## Setup

```bash
cd yolov5
virtualenv --python=python3.8 env
source env/bin/activate
pip install fastapi uvicorn torch torchvision requests wandb tensorboard tqdm pandas opencv-python matplotlib seaborn onnx coremltools onnxruntime
```

```bash
# Dowlnoad weights
cd yolov5
./weights/download_weights.sh
# Test
python detect.py --source 0  --weights yolov5s.pt --conf 0.25
python detect.py --source   path/*.jpg --weights yolov5s.pt --conf 0.25
```

```bash
# Get Dataset
# mkdir -p datasets/dadi/data
cd datasets/dadi/data
curl -L "https://app.roboflow.com/ds/LlUYHr0ytK?key=qBzMwv5jTl" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

```bash
# -- Train
# Custom yolov5s model
# mkdir -p models/custom
#python train.py --img 640 --batch 16 --epochs 100 --data datasets/dadi/yolov5_pytorch_dataset/data.yaml --weights yolov5s.pt --cfg ./models/custom/yolov5s.yaml --name yolov5s_640  --cache
python train.py --img 640 --batch 8 --epochs 200 --data datasets/dadi/yolov5_pytorch_dataset/data.yaml --weights yolov5m.pt --cfg ./models/custom/yolov5m.yaml --name sapera_yolov5m_640  --cache
python train.py --img 640 --batch 8 --epochs 200 --data datasets/dadi/yolov5_pytorch_dataset/data.yaml --weights yolov5s.pt --cfg ./models/custom/yolov5s.yaml --name sapera_yolov5s_640  --cache

# -- Test
python detect.py --img 640 --source 0 --weights runs/train/yolov5s_320/weights/best.pt --conf 0.5
#python detect.py --img 640 --source 0 --weights runs/train/yolov5s_640/weights/best.pt --conf 0.5

python detect.py --img 416 --source datasets/dadi/videos/test1.mp4 --weights runs/train/yolov5s_640/weights/best.pt --conf 0.4 --iou-thres 0.5 --line-thicknes 1

python detect.py --img 416 --source datasets/dadi/videos/test1.mp4 --weights runs/train/yolov5m_640/weights/best.pt --conf 0.6 --hide-conf --iou-thres 0.5 --line-thicknes 1
python detect.py --img 416 --source 0 --weights runs/train/yolov5m_640/weights/best.pt --conf 0.6 --iou-thres 0.5 --line-thicknes 1

# F16
#python models/export.py --weights runs/train/sapera_yolov5m_640/weights/best.pt --img 640 --batch 1 --half --device 0
#python models/export.py --weights runs/train/sapera_yolov5m_640_latest/weights/best.pt --img 640 --batch 1
python models/export.py --weights runs/train/sapera_yolov5m_640_latest/weights/best.pt --include onnx --img 320 --batch 1
python models/export.py --weights runs/train/sapera_yolov5s_640/weights/best.pt --include onnx --img 320 --batch 1


python models/export.py --weights runs/train/sapera_yolov5s_640/weights/best.pt --img 640 --batch 1

python detect.py --img 640 --source datasets/dadi/dadi.jpg --weights runs/train/yolov5m_640/weights/best.pt --conf 0.6 --iou-thres 0.5 --line-thicknes 1


# Pynq z2 https://pynq.readthedocs.io/en/v2.5.1/getting_started/pynq_z2_setup.html
# Openvino https://thenewstack.io/tutorial-accelerate-ai-at-edge-with-onnx-runtime-and-intel-neural-compute-stick-2/
# https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#openvino


# https://thenewstack.io/how-i-built-an-aiot-project-with-intel-ai-vision-x-developer-kit-and-arduino-yun/

# Model conversion
docker pull openvino/ubuntu20_dev:latest
docker pull openvino/ubuntu20_runtime:latest

docker run --rm \
    -u 0 \
    -v /dev:/dev \
    -v /home/visionlab/Documents/annotation-tools/yolov5/deploy:/home/ws \
    --network host \
    -it openvino/ubuntu20_dev:latest \
    /bin/bash  -c "cd /opt/intel/openvino/deployment_tools/model_optimizer && python3 mo.py  --input_model /home/ws/models/best.onnx --output_dir /home/ws/models"

xhost +local:docker && \
docker run --rm --name openvino_ncs2_dev  -it \
        -u 0 \
        --device /dev/video0 \
        --device /dev/dri:/dev/dri \
        --device-cgroup-rule='c 189:* rmw' -v \
        ~/.Xauthority:/root/.Xauthority   \
        -v /dev/bus/usb:/dev/bus/usb \
        -v /home/visionlab/Documents/annotation-tools/yolov5/deploy:/opt/intel/openvino_2021.3.394/ws \
        --network host \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
        -e PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native \
        -v ${XDG_RUNTIME_DIR}/pulse/native:${XDG_RUNTIME_DIR}/pulse/native:Z \
         visiont3lab/openvino-ubuntu18.04 \
        /bin/bash

```