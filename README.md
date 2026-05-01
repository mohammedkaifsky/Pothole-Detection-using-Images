# Pothole Detection (YOLOv8) using your dataset

This project uses your dataset at `C:/Users/acerl/Downloads/archive/Dataset`.

Important: your current dataset is **classification format** (`Normal` vs `Pothole` folders), so this project trains a **YOLOv8 classifier** first.

## 1) Setup

```bash
cd C:/Users/acerl/pothole-yolov8
python -m pip install -r requirements.txt
```

## 2) Train

```bash
python train.py
```

Best model will be saved at:
`runs_cls/pothole_cls/weights/best.pt`

## 3) Evaluate on test split

```bash
python evaluate.py
```

Note: your current dataset has no images in `test/`, so the script auto-evaluates on `val/`.

## 4) Predict on one image

```bash
python predict_image.py --source "C:/path/to/image.jpg" --save
```

## 5) Predict on video or webcam

Video:
```bash
python predict_video.py --source "C:/path/to/video.mp4"
```

Webcam:
```bash
python predict_video.py --source 0
```

## 6) Website interface (upload photo)

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run the web app:
```bash
python app.py
```

Open:
`http://127.0.0.1:5000`

Upload a road image and the app will return:
- `Pothole detected` or `No pothole detected`
- predicted class
- confidence score

## Note on YOLOv8 classification data argument

For classification, Ultralytics expects `data` to be a dataset directory, not a YAML file.
This project already uses:
`C:/Users/acerl/Downloads/archive/Dataset`

## Upgrade to true object detection (bounding boxes)

To get pothole location boxes, annotate images with tools like Roboflow/LabelImg and export in YOLO detection format:

```
images/train, images/val, images/test
labels/train, labels/val, labels/test
```

Then train with `yolov8n.pt` using detection mode.
