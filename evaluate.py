from ultralytics import YOLO
from pathlib import Path

DEFAULT_WEIGHTS_PATHS = (
    Path("runs_cls/pothole_cls/weights/best.pt"),
    Path("runs/classify/runs_cls/pothole_cls/weights/best.pt"),
)
DATASET_DIR = "C:/Users/acerl/Downloads/archive/Dataset"
CLASSES = ("Normal", "Pothole")


def split_has_all_classes(root: Path, split: str) -> bool:
    split_dir = root / split
    if not split_dir.exists():
        return False

    for class_name in CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            return False
        if not any(class_dir.glob("*.*")):
            return False
    return True


def choose_split() -> str:
    root = Path(DATASET_DIR)
    if split_has_all_classes(root, "test"):
        return "test"
    if split_has_all_classes(root, "val"):
        return "val"
    if split_has_all_classes(root, "train"):
        return "train"
    raise RuntimeError(
        "No valid split found with both classes (Normal and Pothole). "
        "Please check dataset folders."
    )


def resolve_weights_path() -> Path:
    for path in DEFAULT_WEIGHTS_PATHS:
        if path.exists():
            return path

    candidates = []
    candidates.extend(Path("runs_cls").glob("*/weights/best.pt"))
    candidates.extend(Path("runs/classify/runs_cls").glob("*/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No trained weights found. Run `python train.py` first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    weights_path = resolve_weights_path()
    model = YOLO(str(weights_path))
    split = choose_split()
    metrics = model.val(data=DATASET_DIR, split=split)
    print(f"Using weights: {weights_path}")
    print(f"Evaluated split: {split}")
    print("Top-1 accuracy:", metrics.top1)
    print("Top-5 accuracy:", metrics.top5)


if __name__ == "__main__":
    main()
