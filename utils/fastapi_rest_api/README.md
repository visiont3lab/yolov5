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
python models/export.py --weights runs/train/sapera_yolov5m_640_latest/weights/best.pt --img 640 --batch 1
python models/export.py --weights runs/train/sapera_yolov5s_640/weights/best.pt --img 640 --batch 1

python detect.py --img 640 --source datasets/dadi/dadi.jpg --weights runs/train/yolov5m_640/weights/best.pt --conf 0.6 --iou-thres 0.5 --line-thicknes 1
```