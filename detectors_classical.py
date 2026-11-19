from pathlib import Path

import cv2 as cv
import numpy as np

from vision_utils import drawCircle, readImage


# =====================================================================
# LAB 1: NAIWNE WYKRYWANIE CZERWONEJ KULKI (PETLA PO PIKSELACH)
# =====================================================================

# algorytm z lab1: petla po pikselach + zal ze na ekranie jest jedna czerwona kulka
def detectRedBallNaive(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    # UWAGA: w OpenCV/NumPy kolejnosc wymiarow to zawsze (height, width, channels), czyli (wiersze, kolumny)
    height, width, _ = image.shape

    min_x, max_x = width, 0
    min_y, max_y = height, 0

    red_pixels = []

    # celowo bez resize: na pelnej klatce z kamery widac realny koszt petli for w czystym Pythonie
    for y in range(height):
        for x in range(width):
            blue, green, red = image[y, x]
            if red > 150 and green < 50 and blue < 50:  # czerwony piksel
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
        return image

    # bounding box:
    bounding_box_width = max_x - min_x + 1
    bounding_box_height = max_y - min_y + 1
    bounding_box_area = bounding_box_width * bounding_box_height

    # pilot params
    fill_ratio = pixel_count / bounding_box_area
    is_symmetric = 0.85 <= bounding_box_width / bounding_box_height <= 1.15

    # kulka na pewno musi: byc symetryczna i wypelniac ok. pi/4 kwadratu
    if not 0.70 <= fill_ratio <= 0.85 or not is_symmetric:
        return image

    # key params
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    radius = (bounding_box_width + bounding_box_height) / 4.0

    # test rownania kola: (x - center_x)**2 + (y - center_y)**2 <= radius**2
    pixels_inside_circle = 0
    for x, y in red_pixels:
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
            pixels_inside_circle += 1

    if pixels_inside_circle / pixel_count >= 0.95:
        drawCircle(image, center_x, center_y, radius, f"Kulka czerwona (fill={fill_ratio:.2f})")

    return image


# =====================================================================
# LAB 2: WYKRYWANIE CZERWONEJ KULKI (SIMPLE BLOB DETECTOR)
# =====================================================================

def createBlobDetector() -> cv.SimpleBlobDetector:
    # tworzymy detektor raz zamiast budowac go od nowa przy kazdej klatce kamery
    blob_parameters = cv.SimpleBlobDetector_Params()

    blob_parameters.filterByColor = True
    blob_parameters.blobColor = 0  # szukamy czarnych plam (0) na odwroconej masce

    blob_parameters.filterByArea = True
    blob_parameters.minArea = 100
    blob_parameters.maxArea = 5000000

    blob_parameters.filterByCircularity = True
    blob_parameters.minCircularity = 0.80

    blob_parameters.filterByConvexity = True
    blob_parameters.minConvexity = 0.90

    blob_parameters.filterByInertia = True
    blob_parameters.minInertiaRatio = 0.85

    return cv.SimpleBlobDetector_create(blob_parameters)


BLOB_DETECTOR = createBlobDetector()


def detectRedBallBlob(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    # zakres czerwieni w formacie BGR (Blue, Green, Red)
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([50, 50, 255])

    # cv.inRange w jednym kroku (w C++) zastepuje podwojna petle for z Lab 1
    red_mask = cv.inRange(image, lower_red, upper_red)
    inverted_mask = cv.bitwise_not(red_mask)  # czerwony -> czarny (0), tlo -> biale (255)

    keypoints = BLOB_DETECTOR.detect(inverted_mask)

    for keypoint in keypoints:
        center_x, center_y = keypoint.pt
        radius = keypoint.size / 2.0  # keypoint.size to srednica, wiec dzielimy przez 2
        drawCircle(image, center_x, center_y, radius, "Kulka czerwona (Blob)", color=(0, 255, 0))

    return image


def convertToGrayscale(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # wracamy do 3 kanalow (BGR), zeby pozniej moc rysowac kolorowy tekst HUD na obrazie
    return cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)


# =====================================================================
# LAB 4: TRANSFORMATA HOUGHA I WYKRYWANIE GRZBIETOW (MACIERZ HESSEGO)
# =====================================================================

def computeHessianRidges(gray_image: np.ndarray, ridge_sigma: float = 2.0) -> np.ndarray:
    """
    Wykrywanie grzbietow (Ridge Detection) z uzyciem drugich pochodnych (Macierz Hessego):
    1. Rozmycie Gaussa usuwa szum i ustawia skale (grubosc szukanej obreczy).
    2. Pierwsze pochodne Sobela (Ix, Iy) -> drugie pochodne (Ixx, Iyy, Ixy).
    3. Wartosci wlasne macierzy Hessego 2x2 wykrywaja szczyt linii niezaleznie od jej kata obrotu.
    """
    float_image = gray_image.astype(np.float32) / 255.0
    kernel_size = int(2 * np.ceil(3 * ridge_sigma) + 1)
    smoothed_image = cv.GaussianBlur(float_image, (kernel_size, kernel_size), sigmaX=ridge_sigma, sigmaY=ridge_sigma)

    # pierwsze pochodne kierunkowe (gradienty)
    derivative_x = cv.Sobel(smoothed_image, cv.CV_32F, 1, 0, ksize=3)
    derivative_y = cv.Sobel(smoothed_image, cv.CV_32F, 0, 1, ksize=3)

    # drugie pochodne czastkowe (elementy macierzy Hessego)
    derivative_xx = cv.Sobel(derivative_x, cv.CV_32F, 1, 0, ksize=3)
    derivative_yy = cv.Sobel(derivative_y, cv.CV_32F, 0, 1, ksize=3)
    derivative_xy = cv.Sobel(derivative_x, cv.CV_32F, 0, 1, ksize=3)

    # wartosci wlasne macierzy 2x2 liczone wektorowo dla wszystkich pikseli naraz
    matrix_trace = derivative_xx + derivative_yy
    matrix_difference = derivative_xx - derivative_yy
    discriminant = np.sqrt(matrix_difference * matrix_difference + 4.0 * (derivative_xy * derivative_xy))

    eigenvalue_1 = 0.5 * (matrix_trace + discriminant)
    eigenvalue_2 = 0.5 * (matrix_trace - discriminant)

    # modul wykrywa zarowno jasne grzbiety na ciemnym tle, jak i ciemne obrecze na jasnym
    ridge_strength = np.maximum(np.abs(eigenvalue_1), np.abs(eigenvalue_2))
    normalized_ridges = cv.normalize(ridge_strength, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # odcinamy slaby szum tla
    _, filtered_ridges = cv.threshold(normalized_ridges, 35, 255, cv.THRESH_TOZERO)
    return filtered_ridges


def classifyBallOrRing(
    gray_image: np.ndarray,
    ridge_map: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> str:
    # porownujemy srodek wykrytego okregu z jego obwodem, zeby odroznic pelna kulke od pustej obreczy
    height, width = gray_image.shape[:2]

    # maska samego srodka (0 .. 0.5 * radius)
    inner_mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(inner_mask, (center_x, center_y), max(2, int(radius * 0.5)), 255, -1)

    # maska pierscienia na obwodzie (0.85 * radius .. 1.10 * radius)
    ring_mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(ring_mask, (center_x, center_y), int(radius * 1.10), 255, -1)
    cv.circle(ring_mask, (center_x, center_y), int(radius * 0.85), 0, -1)

    mean_inner_intensity = cv.mean(gray_image, mask=inner_mask)[0]
    mean_ring_intensity = cv.mean(gray_image, mask=ring_mask)[0]
    intensity_difference = abs(mean_inner_intensity - mean_ring_intensity)

    # strefa przejsciowa (0.45 * radius .. 0.85 * radius): w obreczy widac tu wewnetrzna scianke
    transition_mask = np.zeros((height, width), dtype=np.uint8)
    cv.circle(transition_mask, (center_x, center_y), int(radius * 0.85), 255, -1)
    cv.circle(transition_mask, (center_x, center_y), int(radius * 0.45), 0, -1)
    transition_ridge_strength = cv.mean(ridge_map, mask=transition_mask)[0]

    if intensity_difference > 22.0 or transition_ridge_strength > 45.0:
        return "Obrecz"
    return "Kulka"


def detectCirclesHough(
    source: str | Path | np.ndarray,
    hough_param_1: int = 100,
    hough_param_2: int = 38,
    ridge_sigma: float = 2.0,
) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, (9, 9), 2)
    ridge_map = computeHessianRidges(gray_image, ridge_sigma=ridge_sigma)

    circles = cv.HoughCircles(
        blurred_image,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=hough_param_1,
        param2=hough_param_2,
        minRadius=20,
        maxRadius=300,
    )

    if circles is not None:
        rounded_circles = np.uint16(np.around(circles))
        for circle in rounded_circles[0, :]:
            center_x = int(circle[0])
            center_y = int(circle[1])
            radius = int(circle[2])

            object_type = classifyBallOrRing(gray_image, ridge_map, center_x, center_y, radius)
            circle_color = (0, 255, 0) if object_type == "Kulka" else (255, 200, 0)
            drawCircle(image, center_x, center_y, radius, f"{object_type} (r={radius}px)", color=circle_color)

    return image


def detectRidgesOnly(source: str | Path | np.ndarray, ridge_sigma: float = 2.0) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ridge_map = computeHessianRidges(gray_image, ridge_sigma=ridge_sigma)
    return cv.applyColorMap(ridge_map, cv.COLORMAP_INFERNO)


def detectRidgesAndHough(
    source: str | Path | np.ndarray,
    hough_param_1: int = 100,
    hough_param_2: int = 38,
    ridge_sigma: float = 2.0,
) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ridge_map = computeHessianRidges(gray_image, ridge_sigma=ridge_sigma)
    smoothed_ridges = cv.GaussianBlur(ridge_map, (5, 5), 1.5)

    circles = cv.HoughCircles(
        smoothed_ridges,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=hough_param_1,
        param2=max(15, hough_param_2 - 8),
        minRadius=20,
        maxRadius=300,
    )

    output_image = (image.astype(np.float32) * 0.5).astype(np.uint8)
    output_image[:, :, 0] = np.clip(output_image[:, :, 0].astype(np.int32) + ridge_map, 0, 255).astype(np.uint8)
    output_image[:, :, 1] = np.clip(output_image[:, :, 1].astype(np.int32) + (ridge_map // 2), 0, 255).astype(np.uint8)

    if circles is not None:
        rounded_circles = np.uint16(np.around(circles))
        for circle in rounded_circles[0, :]:
            center_x = int(circle[0])
            center_y = int(circle[1])
            radius = int(circle[2])
            drawCircle(output_image, center_x, center_y, radius, f"Grzbiet Obreczy (r={radius}px)")

    return output_image