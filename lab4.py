import sys
from pathlib import Path

import cv2
import numpy as np

from lab1 import Lab1
from lab2 import Lab2
from lab3 import Lab3

# LAB4: Hough + Ridge Detection
class Lab4:
    def __init__(self):
        self.hough_param1 = 100
        self.hough_param2 = 38
        self.ridge_sigma = 2.0

    def compute_hessian_ridges(self, gray_img: np.ndarray) -> np.ndarray:
        img_float = gray_img.astype(np.float32) / 255.0
        ksize = int(2 * np.ceil(3 * self.ridge_sigma) + 1)
        smoothed = cv2.GaussianBlur(img_float, (ksize, ksize), sigmaX=self.ridge_sigma, sigmaY=self.ridge_sigma)

        Ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

        Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)

        trace = Ixx + Iyy
        diff = Ixx - Iyy
        discriminant = np.sqrt(diff * diff + 4.0 * (Ixy * Ixy))

        lambda1 = 0.5 * (trace + discriminant)
        lambda2 = 0.5 * (trace - discriminant)

        ridge_strength = np.maximum(np.abs(lambda1), np.abs(lambda2))

        ridge_norm = cv2.normalize(ridge_strength, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, ridge_binary = cv2.threshold(ridge_norm, 35, 255, cv2.THRESH_TOZERO)
        return ridge_binary

    def classify_ball_vs_rim(self, gray_frame: np.ndarray, ridge_map: np.ndarray, cx: int, cy: int, r: int) -> str:
        h, w = gray_frame.shape[:2]

        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_inner, (cx, cy), max(2, int(r * 0.5)), 255, -1)

        mask_ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_ring, (cx, cy), int(r * 1.10), 255, -1)
        cv2.circle(mask_ring, (cx, cy), int(r * 0.85), 0, -1)

        mean_inner = cv2.mean(gray_frame, mask=mask_inner)[0]
        mean_ring = cv2.mean(gray_frame, mask=mask_ring)[0]
        intensity_diff = abs(mean_inner - mean_ring)

        mask_transition = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_transition, (cx, cy), int(r * 0.85), 255, -1)
        cv2.circle(mask_transition, (cx, cy), int(r * 0.45), 0, -1)
        transition_ridge = cv2.mean(ridge_map, mask=mask_transition)[0]

        if intensity_diff > 22.0 or transition_ridge > 45.0:
            return "Ring"
        else:
            return "Ball"

    def callback_hough(self, frame):
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        ridge_map = self.compute_hessian_ridges(gray)

        circles = cv2.HoughCircles(
            blurred, # 8-bitowy obraz w skali szarosci
            cv2.HOUGH_GRADIENT, # metoda wykorzystujaca gradienty Sobela, by glosowac na srodki (a, b)
            dp=1.2, # odwrotnosc rozdzielczosci akumulatora (1.2 przyspiesza obliczenia)
            minDist=80,  # minimalna odleglosc w pikselach miedzy srodkami dwoch wykrytych okregow
            param1=self.hough_param1, # gorny prog dla detektora krawedzi Canny'ego (dolny jest 2x mniejszy)
            param2=self.hough_param2, # prog glosow w akumulatorze (im wyzszy, tym pewniejsze okregi)
            minRadius=20, # minimalny promien szukanego obiektu w pikselach 
            maxRadius=300 # maksymalny promien szukanego obiektu w pikselach 
        )

        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))

            for circle in circles_rounded[0, :]:
                cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])

                obj_type = self.classify_ball_vs_rim(gray, ridge_map, cx, cy, r)
                color = (0, 255, 0) if "Ball" in obj_type else (255, 200, 0)

                cv2.circle(output, (cx, cy), r, color, 3)
                cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

                label = f"{obj_type} (r={r}px)"
                cv2.putText(output, label, (max(10, cx - r), max(30, cy - r - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        return output

    def callback_ridges_only(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ridges = self.compute_hessian_ridges(gray)
        return cv2.applyColorMap(ridges, cv2.COLORMAP_INFERNO)

    def callback_ridges_plus_hough(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ridges = self.compute_hessian_ridges(gray)
        ridges_smooth = cv2.GaussianBlur(ridges, (5, 5), 1.5)

        circles = cv2.HoughCircles(
            ridges_smooth,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=self.hough_param1,
            param2=max(15, self.hough_param2 - 8),
            minRadius=20,
            maxRadius=300
        )

        output = (frame.astype(np.float32) * 0.5).astype(np.uint8)
        output[:, :, 0] = np.clip(output[:, :, 0].astype(np.int32) + ridges, 0, 255).astype(np.uint8)
        output[:, :, 1] = np.clip(output[:, :, 1].astype(np.int32) + (ridges // 2), 0, 255).astype(np.uint8)

        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            for circle in circles_rounded[0, :]:
                cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
                cv2.circle(output, (cx, cy), r, (0, 255, 255), 3)
                cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(output, f"Grzbiet Obreczy (r={r}px)", (max(10, cx - r), max(30, cy - r - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        return output

    def get_hud_info(self) -> str:
        return f"Parametry Lab4: Hough prog [w/s]={self.hough_param2} | Grzbiet sigma [e/d]={self.ridge_sigma:.1f}"

    def handle_key(self, key: int):
        if key == ord('w'):
            self.hough_param2 = min(120, self.hough_param2 + 5)
        elif key == ord('s'):
            self.hough_param2 = max(15, self.hough_param2 - 5)
        elif key == ord('e'):
            self.ridge_sigma = min(6.0, self.ridge_sigma + 0.5)
        elif key == ord('d'):
            self.ridge_sigma = max(1.0, self.ridge_sigma - 0.5)

    def main(self):
        lab1 = Lab1()
        lab2 = Lab2()
        lab3 = Lab3()

        callbacks = [
            ("Hough (Kulka vs Obrecz)", self.callback_hough),
            ("Detekcja Grzbietow", self.callback_ridges_only),
            ("Grzbiety + Hough", self.callback_ridges_plus_hough),
            ("Lab 1", lab1.process_frame),
            ("Lab 2", lab2.process_frame),
            ("Lab 3(greyscale)", lab3.callback_grayscale),
            ("Brak", lambda f: f),
        ]

        print("Sterowanie:")
        print("  [c]   - Zmiana algorytmu (Lab 4A -> Lab 4B -> Lab 4C -> Lab 1 -> Lab 2 -> Lab 3)")
        print("  [b]   - Bufor cykliczny (30 ramek) <-> Bez bufora")
        print("  [w/s] - Prog akumulatora Hougha param2 (+5 / -5) [czulosc okregow]")
        print("  [e/d] - Skala grzbietow Sigma (+0.5 / -0.5) [grubosc szukanej obreczy]")
        print("  [+/-] - Zwieksz / zmniejsz opoznienie (+10 ms / -10 ms)")
        print("  [q]   - Wyjscie")

        lab3.run_camera(
            callbacks=callbacks,
            window_title="Lab 4",
            cb_idx=0,
            extra_info_fn=self.get_hud_info,
            extra_key_fn=self.handle_key
        )

if __name__ == "__main__":
    lab4 = Lab4()
    lab4.main()