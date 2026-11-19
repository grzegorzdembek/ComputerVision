import random
from pathlib import Path

import cv2 as cv
import numpy as np


def captureBackgroundFrames(output_directory: Path, target_count: int = 10) -> list[np.ndarray]:
    output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv.VideoCapture(0)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    captured_frames: list[np.ndarray] = []

    if not capture.isOpened():
        print("Nie wykryto kamery - wygeneruje syntetyczne tla zastepcze.")
        return createSyntheticBackgrounds(target_count)

    print("=================================================================")
    print(" KROK 1: ZBIERANIE TLA Z KAMERY")
    print(f"  [SPACJA] - Zrob zdjecie")
    print("  [ENTER]  - Koniec")
    print("=================================================================")

    while True:
        is_read, frame = capture.read()
        if not is_read:
            continue

        preview_frame = frame.copy()
        status_text = f"Zebrane tla: {len(captured_frames)}/{target_count} | [SPACJA] = Zapisz | [ENTER] = Dalej"
        cv.rectangle(preview_frame, (5, 5), (635, 40), (0, 0, 0), -1)
        cv.putText(preview_frame, status_text, (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv.LINE_AA)
        cv.imshow("Lab 5 - Zbieranie tla", preview_frame)

        pressed_key = cv.waitKey(1) & 0xFF
        if pressed_key == 32:  # SPACJA
            file_path = output_directory / f"background_{len(captured_frames):02d}.jpg"
            cv.imwrite(str(file_path), frame)
            captured_frames.append(frame.copy())
            print(f"Zapisano tlo: {file_path.name}")
        elif pressed_key in (13, ord("q")):  # ENTER lub q
            break

    capture.release()
    cv.destroyAllWindows()

    if len(captured_frames) == 0:
        print("Nie zapisano zadnego zdjecia z kamery - uzywam tel syntetycznych.")
        return createSyntheticBackgrounds(target_count)

    return captured_frames

def createSyntheticBackgrounds(count: int, frame_width: int = 640, frame_height: int = 480) -> list[np.ndarray]:
    """Awaryjny generator tel (szum + gradienty), jesli kamera jest odlaczona."""
    backgrounds: list[np.ndarray] = []
    for _ in range(count):
        base_color = np.random.randint(40, 210, size=(1, 1, 3), dtype=np.uint8)
        background = np.full((frame_height, frame_width, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(-25, 25, size=(frame_height, frame_width, 3), dtype=np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        backgrounds.append(background)
    return backgrounds

def drawRedBall(image: np.ndarray) -> tuple[np.ndarray, tuple[int, float, float, float, float]]:
    frame_height, frame_width = image.shape[:2]
    result_image = image.copy()

    radius = random.randint(18, 85)
    margin = radius + 5
    center_x = random.randint(margin, frame_width - margin)
    center_y = random.randint(margin, frame_height - margin)

    red_value = random.randint(165, 255)
    green_value = random.randint(0, 55)
    blue_value = random.randint(0, 55)

    patch_size = radius * 2 + 1
    grid_y, grid_x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance_from_center = np.sqrt(grid_x**2 + grid_y**2)
    circle_mask = distance_from_center <= radius

    light_offset_x = random.uniform(-0.35, 0.35) * radius
    light_offset_y = random.uniform(-0.35, 0.35) * radius
    light_distance = np.sqrt((grid_x - light_offset_x) ** 2 + (grid_y - light_offset_y) ** 2)
    shading_factor = np.clip(1.15 - 0.45 * (light_distance / max(1, radius)), 0.55, 1.25)

    ball_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
    ball_patch[:, :, 0] = np.clip(blue_value * shading_factor, 0, 255)
    ball_patch[:, :, 1] = np.clip(green_value * shading_factor, 0, 255)
    ball_patch[:, :, 2] = np.clip(red_value * shading_factor, 0, 255)

    top_y = center_y - radius
    left_x = center_x - radius
    roi = result_image[top_y : top_y + patch_size, left_x : left_x + patch_size].astype(np.float32)
    roi[circle_mask] = ball_patch[circle_mask]
    result_image[top_y : top_y + patch_size, left_x : left_x + patch_size] = roi.astype(np.uint8)

    if random.random() < 0.7:
        result_image = cv.GaussianBlur(result_image, (3, 3), 0)

    box_width = 2.0 * radius
    box_height = 2.0 * radius

    x_center_norm = center_x / frame_width
    y_center_norm = center_y / frame_height
    width_norm = box_width / frame_width
    height_norm = box_height / frame_height

    yolo_label = (0, x_center_norm, y_center_norm, width_norm, height_norm)
    return result_image, yolo_label


def drawDistractorObject(image: np.ndarray) -> np.ndarray:
    frame_height, frame_width = image.shape[:2]
    result_image = image.copy()

    size = random.randint(25, 80)
    center_x = random.randint(size + 5, frame_width - size - 5)
    center_y = random.randint(size + 5, frame_height - size - 5)

    if random.random() < 0.5:
        cv.rectangle(
            result_image,
            (center_x - size, center_y - size),
            (center_x + size, center_y + size),
            (10, 10, random.randint(170, 255)),
            -1,
        )
    else:
        wrong_color = (random.randint(170, 255), 30, 20) if random.random() < 0.5 else (30, random.randint(170, 255), 20)
        cv.circle(result_image, (center_x, center_y), size, wrong_color, -1)

    return result_image

def generateDatasetSplit(
    dataset_root: Path,
    split_name: str,
    backgrounds: list[np.ndarray],
    image_count: int,
    preview_directory: Path | None = None,
) -> None:
    images_directory = dataset_root / "images" / split_name
    labels_directory = dataset_root / "labels" / split_name
    images_directory.mkdir(parents=True, exist_ok=True)
    labels_directory.mkdir(parents=True, exist_ok=True)

    for index in range(image_count):
        base_background = random.choice(backgrounds).copy()

        if random.random() < 0.5:
            base_background = cv.flip(base_background, 1)

        label_lines: list[str] = []

        if random.random() < 0.85:
            ball_count = 2 if random.random() < 0.20 else 1
            generated_image = base_background
            for _ in range(ball_count):
                generated_image, (class_id, x_norm, y_norm, w_norm, h_norm) = drawRedBall(generated_image)
                label_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
        else:
            generated_image = drawDistractorObject(base_background) if random.random() < 0.6 else base_background

        stem_name = f"{split_name}_{index:03d}"
        image_path = images_directory / f"{stem_name}.jpg"
        label_path = labels_directory / f"{stem_name}.txt"

        cv.imwrite(str(image_path), generated_image)
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if preview_directory is not None and index < 3 and len(label_lines) > 0:
            preview_directory.mkdir(parents=True, exist_ok=True)
            preview_image = generated_image.copy()
            frame_height, frame_width = preview_image.shape[:2]
            for line in label_lines:
                _, x_norm, y_norm, w_norm, h_norm = map(float, line.split())
                x1 = int((x_norm - w_norm / 2.0) * frame_width)
                y1 = int((y_norm - h_norm / 2.0) * frame_height)
                x2 = int((x_norm + w_norm / 2.0) * frame_width)
                y2 = int((y_norm + h_norm / 2.0) * frame_height)
                cv.rectangle(preview_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(preview_image, "red_ball (0)", (x1, max(20, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv.imwrite(str(preview_directory / f"preview_{stem_name}.jpg"), preview_image)

def createDatasetYaml(lab_directory: Path, dataset_root: Path) -> Path:
    yaml_path = lab_directory / "dataset.yaml"
    yaml_content = (
        f"path: {dataset_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        f"  0: red_ball\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path

def main() -> None:
    lab_directory = Path(__file__).resolve().parent
    backgrounds_directory = lab_directory / "backgrounds"
    dataset_root = lab_directory / "dataset"
    preview_directory = lab_directory / "preview"

    backgrounds = captureBackgroundFrames(backgrounds_directory, target_count=10)

    print("Generowanie zbiorow train / val / test wraz z etykietami YOLO...")
    generateDatasetSplit(dataset_root, "train", backgrounds, image_count=70, preview_directory=preview_directory)
    generateDatasetSplit(dataset_root, "val", backgrounds, image_count=20)
    generateDatasetSplit(dataset_root, "test", backgrounds, image_count=15)

    yaml_path = createDatasetYaml(lab_directory, dataset_root)

    print("Gotowe")
    print(f"  Zbiór danych: {dataset_root}")
    print(f"  Konfiguracja: {yaml_path}")
    print(f"  Podglad etykiet: {preview_directory}")

if __name__ == "__main__":
    main()