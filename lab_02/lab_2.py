import cv2
import numpy as np

# find red ball 2
class Lab2: # zalozenie ze na ekranie jest tylko jedna czerwona kulka
    def __init__(self, image: str):
        self.image_path = image
        self.image = cv2.imread(image)
        self.height, self.width, self.channels = self.image.shape

    def main(self):
        lower_red = np.array([0, 0, 150])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(self.image, lower_red, upper_red)
        mask_inv = cv2.bitwise_not(mask)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 0

        params.filterByArea = True
        params.minArea = 100
        params.maxArea = self.width * self.height

        params.filterByCircularity = True
        params.minCircularity = 0.85

        params.filterByConvexity = True
        params.minConvexity = 0.90

        params.filterByInertia = True
        params.minInertiaRatio = 0.85

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(mask_inv)

        if len(keypoints) > 0:
            print("Wykryto czerwoną kulkę!")
            output = cv2.drawKeypoints(self.image, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow(f"Wynik - {self.image_path}", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        else:
            print(f"[{self.image_path}] Nie wykryto czerwonej kulki (wykryto coś innego lub brak obiektu).")
            return False


if __name__ == "__main__":
    redball = Lab2(image="redBall.jpg")
    redSquare = Lab2(image="redSquare.jpg")
    redOddity = Lab2(image="redOddity.jpg")

    redball.main()
    redSquare.main()
    redOddity.main()


