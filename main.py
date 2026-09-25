"""
Usage:
    python main.py                      
    python main.py --source 1
^ ^ this one works on the pi, does not work with no argument 


    python main.py --source clip.mp4    
    python main.py --source shot.png    

    press q to end

    trained on 500 images of mobs in Minecraft, using YOLO26n
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).parent / "model" / "weights"
NCNN_PATH = WEIGHTS_DIR / "best_ncnn_model"
MODEL_PATH = NCNN_PATH if NCNN_PATH.exists() else WEIGHTS_DIR / "best.pt"

COLORS = {
    "creeper": (0, 200, 0),
    "skeleton": (220, 220, 220),
    "spider": (0, 0, 200),
    "zombie": (200, 120, 0),
    "enderman": (200, 0, 200),
}


def draw_detections(frame, result):
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = names[int(box.cls[0])]
        score = float(box.conf[0])
        color = COLORS.get(name, (0, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y1, th + baseline + 4)
        cv2.rectangle(frame, (x1, top - th - baseline - 4), (x1 + tw + 4, top), color, -1)
        cv2.putText(frame, label, (x1 + 2, top - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="mminecraft mob detector")
    parser.add_argument("--source", default="0", help='a webcam index (0), "screen", or an image/video path')
    parser.add_argument("--conf", type=float, default=0.4, help="minimum confidence (0-1)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(MODEL_PATH), task="detect")

    is_image = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for result in model.predict(source, conf=args.conf, stream=True, verbose=False):
        frame = draw_detections(result.orig_img.copy(), result)
        cv2.imshow("mob detector", frame)

        if is_image:
            cv2.waitKey(0)
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
