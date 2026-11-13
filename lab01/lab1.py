import cv2

# LAB1: find red ball (naive) 
class Lab1: # zalozenie ze na ekranie jest tylko jedna czerwona kulka
    def __init__(self, image: str = None):
        self.image = cv2.imread(image) if image is not None else None
        if self.image is not None:
            self.height, self.width, self.channels = self.image.shape

    def detect(self, image):
        height, width, _ = image.shape
        min_x = width
        max_x = 0
        min_y = height
        max_y = 0
        red_pixels = []

        for y in range(height):
            for x in range(width):
                b, g, r = image[y, x]
                if r > 150 and g < 50 and b < 50: # czerwony piksel
                    red_pixels.append((x, y))

                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y

        pixel_count = len(red_pixels)
        if pixel_count == 0:
            return "brak", None

        bb_width = max_x - min_x + 1
        bb_height = max_y - min_y + 1
        bb_area = bb_width * bb_height

        fill_ratio = pixel_count / bb_area
        is_symmetric = 0.85 <= bb_width / bb_height <= 1.15

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        radius = (bb_width + bb_height) / 4.0

        info = (min_x, min_y, max_x, max_y, center_x, center_y, radius, fill_ratio)

        # kulka na pewno musi: byc symetryczna i wypelniac ok. pi/4 kwadratu
        if not 0.70 <= fill_ratio <= 0.85 or not is_symmetric:
            return "cos_innego", info

        # test rownania kola: (x-cx)**2 + (y-cy)**2 <= r^2
        pixels_inside_circle = 0
        for x, y in red_pixels:
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                pixels_inside_circle += 1

        if pixels_inside_circle / pixel_count >= 0.95:
            return "kulka", info
        else:
            return "cos_innego", info

    def process_frame(self, frame): # callback pod kamere (lab3)
        output = frame.copy()
        orig_h, orig_w = frame.shape[:2]

        # zmniejszamy ramke robocza, zeby podwojna petla for w Pythonie dzialala plynnie na zywo
        scale = 160.0 / orig_w
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        status, info = self.detect(small_frame)
        if status == "brak":
            return output

        min_x, min_y, max_x, max_y, center_x, center_y, radius, fill_ratio = info
        inv_scale = 1.0 / scale
        rx1, ry1 = int(min_x * inv_scale), int(min_y * inv_scale)
        rx2, ry2 = int(max_x * inv_scale), int(max_y * inv_scale)
        rcx, rcy = int(center_x * inv_scale), int(center_y * inv_scale)
        rr = int(radius * inv_scale)

        if status == "kulka":
            cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.circle(output, (rcx, rcy), rr, (0, 255, 0), 3)
            cv2.putText(output, f"Lab1: Kulka (fill={fill_ratio:.2f})", (rx1, max(25, ry1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        else:
            cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            cv2.putText(output, f"Lab1: Cos innego (fill={fill_ratio:.2f})", (rx1, max(25, ry1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        return output

    def main(self):
        status, _ = self.detect(self.image)
        if status == "brak":
            print("Nie wykryto czerwonych pikseli.")
            return False
        elif status == "kulka":
            print("Wykryto czerwoną kulkę!")
            return True
        else:
            print("Wykryto coś innego.")
            return False


if __name__ == "__main__":
    redball = Lab1(image="redBall.jpg")
    perfectRedball = Lab1(image="perfectRedBall.jpg")
    redSquare = Lab1(image="redSquare.jpg")
    redOddity = Lab1(image="redOddity.jpg")

    redball.main()
    perfectRedball.main()
    redSquare.main()
    redOddity.main()