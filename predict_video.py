import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pothole classification on video/webcam")
    parser.add_argument("--weights", default="runs_cls/pothole_cls/weights/best.pt")
    parser.add_argument(
        "--source",
        default="0",
        help="Video path or webcam index (0 for default webcam)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    model.predict(source=source, conf=args.conf, show=True, save=True)


if __name__ == "__main__":
    main()
