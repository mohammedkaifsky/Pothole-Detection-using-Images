import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pothole/no-pothole on an image")
    parser.add_argument("--weights", default="runs_cls/pothole_cls/weights/best.pt")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--save", action="store_true", help="Save annotated output image")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=0.25, save=args.save)

    probs = results[0].probs
    class_id = int(probs.top1)
    class_name = results[0].names[class_id]
    confidence = float(probs.top1conf)

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
