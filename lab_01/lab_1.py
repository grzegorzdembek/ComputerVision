import cv2 

# find red ball
class Lab1: # zalozenie ze na ekranie jest tylko jedna czerwona kulka
    def __init__(self, image: str):
        self.image = cv2.imread(image)
        self.height, self.width, self.channels = self.image.shape

    def main(self):
        min_x = self.width
        max_x = 0
        min_y = self.height
        max_y = 0
        red_pixels = []
        
        for y in range(self.height):
            for x in range(self.width):
                b,g,r = self.image[y,x]
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
            print("Nie wykryto czerwonych pikseli.")
            return False

        bb_width = max_x - min_x + 1
        bb_height = max_y - min_y + 1

        bb_area = bb_width * bb_height

        fill_ratio = pixel_count / bb_area 
        is_symmetric = 0.85 <= bb_width / bb_height <= 1.15  

        # kulka na pewno musi: byc symetryczna i wypelniac ok. pi/4 kwadratu
        if not 0.70 <= fill_ratio <= 0.85 or not is_symmetric:
            print("Wykryto coś innego.")
            return False

        # test rownania kola: (x-cx)**2 + (y-cy)**2 <= r^2
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        radius = (bb_width + bb_height) / 4.0 
 
        pixels_inside_circle = 0
        for x, y in red_pixels:
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                pixels_inside_circle += 1

        if pixels_inside_circle / pixel_count >= 0.98:
            print("Wykryto czerwoną kulkę!")
            return True         
        else:
            print("Wykryto coś innego.")
            return False
        
if __name__ == "__main__":
    redball = Lab1(image="redBall.jpg")
    redSquare = Lab1(image="redSquare.jpg")
    redOddity = Lab1(image="redOddity.jpg")

    redball.main()
    redSquare.main()
    redOddity.main()


