import time
from pathlib import Path

import cv2 as cv

from camera_stream import CameraGrabber
from detector_yolo import detectRedBallYolo
from detectors_classical import (
    convertToGrayscale,
    detectCirclesHough,
    detectRedBallBlob,
    detectRedBallNaive,
    detectRidgesAndHough,
    detectRidgesOnly,
)
from vision_utils import drawHudOverlay, showImageResult

def runCamera(window_title: str = "Real-Time Computer Vision Sandbox", start_callback_index: int = 0) -> None:
    grabber = CameraGrabber(camera_source=0, buffer_capacity=30, use_buffer=True).start()

    # parametry Lab 4 pozwala modyfikowac je z klawiatury w trakcie dzialania programu
    hough_settings = {
        "hough_param_1": 100,
        "hough_param_2": 38,
        "ridge_sigma": 2.0,
    }

    # kazdy callback przyjmuje 1 argument (frame) i zwraca przetworzony obraz (np.ndarray)
    callbacks = [
        ("Lab 5: YOLO11n (Deep Learning)", detectRedBallYolo),
        (
            "Lab 4A: Hough (Kulka vs Obrecz)",
            lambda frame: detectCirclesHough(
                frame,
                hough_param_1=hough_settings["hough_param_1"],
                hough_param_2=hough_settings["hough_param_2"],
                ridge_sigma=hough_settings["ridge_sigma"],
            ),
        ),
        (
            "Lab 4B: Detekcja Grzbietow (Hessian)",
            lambda frame: detectRidgesOnly(frame, ridge_sigma=hough_settings["ridge_sigma"]),
        ),
        (
            "Lab 4C: Fuzja Grzbiety + Hough",
            lambda frame: detectRidgesAndHough(
                frame,
                hough_param_1=hough_settings["hough_param_1"],
                hough_param_2=hough_settings["hough_param_2"],
                ridge_sigma=hough_settings["ridge_sigma"],
            ),
        ),
        ("Lab 1: Naiwna petla po pikselach", detectRedBallNaive),
        ("Lab 2: SimpleBlobDetector", detectRedBallBlob),
        ("Lab 3: Skala szarosci", convertToGrayscale),
        ("Brak", lambda frame: frame.copy()),
    ]

    callback_index = start_callback_index % len(callbacks)
    artificial_delay_seconds = 0.0

    displayed_frames = 0
    display_fps = 0.0
    dropped_fps = 0.0
    total_dropped = 0
    metrics_start_time = time.perf_counter()

    print("Sterowanie:")
    print("  [c]   - Zmiana algorytmu (lab5 -> lab4 -> lab1 -> lab2 -> lab3 -> Raw)")
    print("  [b]   - Bufor cykliczny (30 ramek) <-> Bez bufora")
    print("  [w/s] - Prog akumulatora Hougha param2 (+5 / -5)")
    print("  [e/d] - Skala grzbietow Sigma (+0.5 / -0.5)")
    print("  [+/-] - Sztuczne opoznienie (+10 ms / -10 ms)")
    print("  [q]   - Wyjscie")

    while True:
        frame_item = grabber.getFrame()
        if frame_item is None:
            time.sleep(0.001)
            continue

        frame, capture_timestamp = frame_item

        callback_name, callback_function = callbacks[callback_index]
        processed_frame = callback_function(frame)

        if artificial_delay_seconds > 0:
            time.sleep(artificial_delay_seconds)

        current_time = time.perf_counter()
        latency_ms = (current_time - capture_timestamp) * 1000.0
        frame_height, frame_width = processed_frame.shape[:2]

        displayed_frames += 1
        elapsed_time = current_time - metrics_start_time
        if elapsed_time >= 0.5:
            display_fps = displayed_frames / elapsed_time
            drops, buffer_fill, buffer_capacity = grabber.getDropStats()
            dropped_fps = drops / elapsed_time
            total_dropped += drops
            displayed_frames = 0
            metrics_start_time = current_time
        else:
            buffer_fill = grabber.ring_buffer.count if grabber.use_buffer else 0
            buffer_capacity = grabber.ring_buffer.capacity

        buffer_mode_text = f"Bufor cykliczny ({buffer_fill}/{buffer_capacity})" if grabber.use_buffer else "Bez bufora"
        info_lines = [
            f"Tryb [b]: {buffer_mode_text}",
            f"Callback [c]: {callback_name} | Opoznienie sztuczne [+/-]: {int(artificial_delay_seconds * 1000)} ms",
            f"Rozdzielczosc: {frame_width}x{frame_height} px | Opoznienie (Latency): {latency_ms:.1f} ms",
            f"Kamera FPS: {grabber.capture_fps:.1f} | Wyswietlane FPS: {display_fps:.1f}",
            f"Gubione FPS: {dropped_fps:.1f}/s (Lacznie zgubiono: {total_dropped})",
            f"Hough param2 [w/s]={hough_settings['hough_param_2']} | Sigma [e/d]={hough_settings['ridge_sigma']:.1f}",
        ]

        drawHudOverlay(processed_frame, info_lines, latency_ms)
        cv.imshow(window_title, processed_frame)

        pressed_key = cv.waitKey(1) & 0xFF
        if pressed_key == ord("q"):
            break
        elif pressed_key == ord("b"):
            grabber.use_buffer = not grabber.use_buffer
        elif pressed_key == ord("c"):
            callback_index = (callback_index + 1) % len(callbacks)
        elif pressed_key in (ord("+"), ord("=")):
            artificial_delay_seconds = min(0.5, artificial_delay_seconds + 0.01)
        elif pressed_key == ord("-"):
            artificial_delay_seconds = max(0.0, artificial_delay_seconds - 0.01)
        elif pressed_key == ord("w"):
            hough_settings["hough_param_2"] = min(120, hough_settings["hough_param_2"] + 5)
        elif pressed_key == ord("s"):
            hough_settings["hough_param_2"] = max(15, hough_settings["hough_param_2"] - 5)
        elif pressed_key == ord("e"):
            hough_settings["ridge_sigma"] = min(6.0, hough_settings["ridge_sigma"] + 0.5)
        elif pressed_key == ord("d"):
            hough_settings["ridge_sigma"] = max(1.0, hough_settings["ridge_sigma"] - 0.5)

    grabber.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # --- Test na pojedynczym pliku (odkomentuj do szybkiego testu) ---
    # test_image_path = Path("redBall.jpg")
    # showImageResult("YOLO11n", detectRedBallYolo(test_image_path))

    runCamera()