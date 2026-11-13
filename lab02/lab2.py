import cv2
import numpy as np

# LAB2: find red ball 2 (blob)
class Lab2: # zalozenie ze na ekranie jest tylko jedna czerwona kulka
    def __init__(self, image: str = None):
        self.image_path = image
        self.image = cv2.imread(image) if image is not None else None

        self.lower_red = np.array([0, 0, 150])
        self.upper_red = np.array([50, 50, 255])

        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 0

        params.filterByArea = True
        params.minArea = 100
        if self.image is not None:
            self.height, self.width, self.channels = self.image.shape
            params.maxArea = self.width * self.height
        else:
            params.maxArea = 5000000

        params.filterByCircularity = True
        params.minCircularity = 0.80

        params.filterByConvexity = True
        params.minConvexity = 0.90

        params.filterByInertia = True
        params.minInertiaRatio = 0.85

        self.detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame):
        mask = cv2.inRange(frame, self.lower_red, self.upper_red)
        mask_inv = cv2.bitwise_not(mask) # red -> black(0), background -> white(255)
        return self.detector.detect(mask_inv)

    def process_frame(self, frame): # callback pod kamere (lab3)
        keypoints = self.detect(frame)
        if len(keypoints) > 0:
            return cv2.drawKeypoints(
                frame,
                keypoints,
                np.array([]),
                (0, 255, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
        return frame

    def main(self):
        keypoints = self.detect(self.image)

        if len(keypoints) > 0:
            print(f"[{self.image_path}] Wykryto czerwoną kulkę!")
            output = self.process_frame(self.image)
            cv2.imshow(f"Wynik - {self.image_path}", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        else:
            print(f"[{self.image_path}] Nie wykryto czerwonej kulki.")
            return False

if __name__ == "__main__":
    redball = Lab2(image="redBall.jpg")
    #redSquare = Lab2(image="redSquare.jpg")
    #redOddity = Lab2(image="redOddity.jpg")

    redball.main()
    #redSquare.main()
    #redOddity.main()