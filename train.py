from ultralytics import YOLO

DATASET_DIR = "C:/Users/acerl/Downloads/archive/Dataset"
MODEL_NAME = "yolov8n-cls.pt"


def main() -> None:
    model = YOLO(MODEL_NAME)
    model.train(
        data=DATASET_DIR,
        epochs=30,
        imgsz=224,
        batch=32,
        project="runs_cls",
        name="pothole_cls",
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
    )


if __name__ == "__main__":
    main()
