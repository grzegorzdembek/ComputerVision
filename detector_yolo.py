from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

from vision_utils import drawBoundingBox, readImage


def loadYoloModel() -> YOLO | None:
    weights_path = Path(__file__).resolve().parent / "models" / "red_ball_yolo11n.pt"
    if not weights_path.exists():
        print(f"[YOLO] Nie znaleziono wag modelu: {weights_path}")
        return None
    return YOLO(str(weights_path))


# ladujemy wagi wytrenowanej sieci raz przy starcie programu
YOLO_MODEL = loadYoloModel()


def detectRedBallYolo(source: str | Path | np.ndarray, confidence_threshold: float = 0.50) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    if YOLO_MODEL is None:
        cv.putText(image, "YOLO: Brak pliku models/red_ball_yolo11n.pt", (20, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image

    # verbose=False wylacza spamowanie konsoli logami przy kazdej pojedynczej klatce kamery
    prediction_results = YOLO_MODEL.predict(image, conf=confidence_threshold, imgsz=416, verbose=False)

    for result in prediction_results:
        for detected_box in result.boxes:
            # wyciagamy wspolrzedne prostokata (x1, y1, x2, y2) z tensora GPU do tablicy NumPy
            box_coordinates = detected_box.xyxy[0].cpu().numpy()
            top_left_x = int(box_coordinates[0])
            top_left_y = int(box_coordinates[1])
            bottom_right_x = int(box_coordinates[2])
            bottom_right_y = int(box_coordinates[3])

            confidence = float(detected_box.conf[0])
            label_text = f"YOLO red_ball ({confidence * 100:.1f}%)"

            drawBoundingBox(image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, label_text)

    return image