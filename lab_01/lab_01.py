import cv2 

# find red ball
class Lab01: # zalozenie ze na ekranie jest tylko jedna czerwona kulka
    def __init__(self, image: str):
        self.image = cv2.imread(image)
        self.height, self.width, self.channels = self.image.shape

    def main(self):
        min_x = self.width
        max_x = 0
        min_y = self.height
        max_y = 0
        pixel_count = 0
        
        for y in range(self.height):
            for x in range(self.width):
                b,g,r = self.image[y,x]
                if r > 150 and g < 50 and b < 50: # czerwony piksel
                    pixel_count += 1
        
                    if x < min_x: 
                        min_x = x
                    if x > max_x: 
                        max_x = x
                    if y < min_y: 
                        min_y = y
                    if y > max_y: 
                        max_y = y
        
        if pixel_count == 0:
            print ("Nie wykryto czerwonych pikseli.")
            return False 

        bb_width = max_x - min_x + 1
        bb_height = max_y - min_y + 1

        bb_area = bb_width * bb_height

        fill_ratio = pixel_count / bb_area 
        is_symmetric = 0.85 <= bb_width / bb_height <= 1.15  
            
        if 0.70 <= fill_ratio <= 0.85 and is_symmetric:
            print("Wykryto czerwoną kulkę.")
            return True            
        else:
            print("Wykryto coś innego.")
            return False
        

if __name__ == "__main__":
    redball = Lab01(image="redBall.jpg")
    redSquare = Lab01(image="redSquare.jpg")
    redOddity = Lab01(image="redOddity.jpg")

    redball.main()
    redSquare.main()
    redOddity.main()


