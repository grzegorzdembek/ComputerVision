from pathlib import Path

import cv2 as cv
import numpy as np


# chce obsluzyc obraz z pliku i ramke z kamery
def readImage(source: str | Path | np.ndarray) -> np.ndarray | None:
    if isinstance(source, np.ndarray):
        # robimy kopie, zeby rysowanie po ramce nie nadpisalo oryginalu w pamieci
        return source.copy()
    if isinstance(source, (str, Path)):
        # OpenCV wymaga zwyklego stringa, dlatego rzutujemy Path na str
        return cv.imread(str(source))
    return None


def drawBoundingBox(
    image: np.ndarray,
    top_left_x: int,
    top_left_y: int,
    bottom_right_x: int,
    bottom_right_y: int,
    label_text: str,
    color: tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    cv.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 3)

    text_x = max(10, top_left_x)
    text_y = max(30, top_left_y - 10)
    text_position = (text_x, text_y)

    # cienka czarna obwodka pod tekstem (grubosc 4) sprawia, ze napis jest czytelny na kazdym tle
    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv.LINE_AA)
    return image


def drawCircle(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    label_text: str,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    # funkcje rysujace w OpenCV wymagaja wspolrzednych calkowitych (int), a nie float
    center_x_int = int(round(center_x))
    center_y_int = int(round(center_y))
    radius_int = int(round(radius))

    # obwodka zoltym flamastrem (domyslnie BGR: 0, 255, 255) oraz czerwony punkt w srodku
    cv.circle(image, (center_x_int, center_y_int), radius_int, color, 3)
    cv.circle(image, (center_x_int, center_y_int), 4, (0, 0, 255), -1)

    # pozycja napisu: tuz nad obiektem, ale z blokada, zeby tekst nie uciekl poza ekran
    text_x = max(10, center_x_int - radius_int)
    text_y = max(30, center_y_int - radius_int - 10)
    text_position = (text_x, text_y)

    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv.LINE_AA)
    return image


def drawHudOverlay(frame: np.ndarray, info_lines: list[str], latency_ms: float) -> np.ndarray:
    box_height = 15 + len(info_lines) * 25
    overlay = frame.copy()
    cv.rectangle(overlay, (5, 5), (720, box_height), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for index, line in enumerate(info_lines):
        if latency_ms < 100:
            text_color = (0, 255, 0)
        elif latency_ms < 300:
            text_color = (0, 165, 255)
        else:
            text_color = (0, 0, 255)

        cv.putText(
            frame,
            line,
            (15, 28 + index * 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.52,
            text_color,
            1,
            cv.LINE_AA,
        )

    return frame


def showImageResult(window_title: str, image: np.ndarray | None) -> None:
    # funkcja pomocnicza do szybkiego testowania dowolnego algorytmu na statycznym pliku .jpg
    if image is None:
        print(f"[{window_title}] Nie udalo sie wczytac obrazu.")
        return

    cv.imshow(window_title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()