# lab1: naiwne wykrywanie czerwonej kulki (petla po pikselach + rownanie kola)
# lab2: wykrywanie czerwonej kulki za pomoca maski koloru i SimpleBlobDetector
# lab3: wielowatkowy odczyt kamery (CameraGrabber + ThreadSafeRingBuffer) + metryki
# lab4: transformata Hougha (kulka vs obrecz) + wykrywanie grzbietow (macierz Hessego)
# lab5: znajdz czerwona kulke "AI"

from ultralytics import YOLO
import threading
import time
from pathlib import Path

import cv2 as cv
import numpy as np

# =====================================================================
# FUNKCJE POMOCNICZE (WEJSCIE / RYSOWANIE / KONFIGURACJA)
# =====================================================================

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

    # cienka czarna obwodka pod tekstem (grubosc 4) sprawia, ze napis jest czytelny na kazdym tle
    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, label_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv.LINE_AA)
    return image


def showImageResult(window_title: str, image: np.ndarray | None) -> None:
    # funkcja pomocnicza do szybkiego testowania dowolnego algorytmu na statycznym pliku .jpg
    if image is None:
        print(f"[{window_title}] Nie udalo sie wczytac obrazu.")
        return

    cv.imshow(window_title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# =====================================================================
# LAB 1: NAIWNE WYKRYWANIE CZERWONEJ KULKI (PETLA PO PIKSELACH)
# =====================================================================

# algorytm z lab1: petla po pikselach + zal ze na ekranie jest jedna czerwona kulka
def detectRedBallNaive(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    # UWAGA: w OpenCV/NumPy kolejnosc wymiarow to zawsze (height, width, channels), czyli (wiersze, kolumny)!
    # zamiana ich miejscami na prostokatnym obrazie z kamery powoduje IndexError
    height, width, _ = image.shape

    min_x, max_x = width, 0
    min_y, max_y = height, 0

    red_pixels = []

    # celowo bez resize: na pelnej klatce z kamery zobaczysz ogromne opoznienie (latency)
    # to swietnie pokazuje, dlaczego w CV unikamy czystych petli for w Pythonie
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
        # tu rysujemy "obwodke zoltym flamastrem" wspolna funkcja drawCircle
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


# staly obiekt detektora wspoldzielony miedzy wywolaniami funkcji detectRedBallBlob
BLOB_DETECTOR = createBlobDetector()


def detectRedBallBlob(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    # zakres czerwieni w formacie BGR (Blue, Green, Red)
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([50, 50, 255])

    # cv.inRange w jednym kroku (w C++) zastepuje nasza podwojna petle for z Lab 1
    red_mask = cv.inRange(image, lower_red, upper_red)
    inverted_mask = cv.bitwise_not(red_mask)  # czerwony -> czarny (0), tlo -> biale (255)

    keypoints = BLOB_DETECTOR.detect(inverted_mask)

    for keypoint in keypoints:
        center_x, center_y = keypoint.pt
        radius = keypoint.size / 2.0  # keypoint.size to srednica, wiec dzielimy przez 2
        drawCircle(image, center_x, center_y, radius, "Kulka czerwona (Blob)", color=(0, 255, 0))

    return image


# =====================================================================
# LAB 3: BUFOR CYKLICZNY I NIEBLOKUJACY ODCZYT KAMERY
# =====================================================================

def convertToGrayscale(source: str | Path | np.ndarray) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # wracamy do 3 kanalow (BGR), zeby pozniej moc rysowac kolorowy tekst HUD na obrazie
    return cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)


class ThreadSafeRingBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.buffer = [None] * self.capacity
        self.head = 0  # odczyt - watek Konsumenta (runCamera)
        self.tail = 0  # zapis - watek Producenta (CameraGrabber)
        self.count = 0
        self.dropped_frames = 0
        self.lock = threading.Lock()

    def put(self, item) -> None:  # watek Producenta
        with self.lock:
            if self.count == self.capacity:
                # bufor pelny: nadpisujemy najstarsza ramke i przesuwamy glowe (modulo zamyka petle)
                self.head = (self.head + 1) % self.capacity
                self.dropped_frames += 1
            else:
                self.count += 1

            self.buffer[self.tail] = item  # zapisana najnowsza ramka
            self.tail = (self.tail + 1) % self.capacity

    def get(self):  # watek Konsumenta
        with self.lock:
            if self.count == 0:
                return None

            item = self.buffer[self.head]  # odczytana najstarsza ramka
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
            return item

    def getStatsAndResetDrops(self) -> tuple[int, int, int]:  # info do wyswietlenia na ekranie
        with self.lock:
            drops = self.dropped_frames
            self.dropped_frames = 0
            return drops, self.count, self.capacity


class CameraGrabber:  # nie blokuje odczytywania ramek
    def __init__(
        self,
        camera_source: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        buffer_capacity: int = 30,
        use_buffer: bool = True,
    ):
        self.capture = cv.VideoCapture(camera_source)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        # zmniejszamy wewnetrzny bufor sterownika OpenCV do 1 klatki, aby uniknac ukrytych opoznien
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.use_buffer = use_buffer
        self.ring_buffer = ThreadSafeRingBuffer(buffer_capacity)

        # zmienne dla trybu "bez bufora" (zawsze trzymamy tylko najnowsza klatke)
        self.latest_frame = None
        self.latest_lock = threading.Lock()
        self.unbuffered_drops = 0

        self.is_running = False
        self.capture_thread = None
        self.capture_fps = 0.0

    def start(self):  # uruchamia zbieranie klatek w tle
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._captureLoop, daemon=True)
        self.capture_thread.start()
        return self

    def _captureLoop(self) -> None:  # watek w tle: odbiera klatki tak szybko jak pozwala kamera
        frames_captured = 0
        start_time = time.perf_counter()

        while self.is_running:
            is_frame_read, frame = self.capture.read()
            if not is_frame_read:
                time.sleep(0.005)
                continue

            # zapisujemy dokladny czas pobrania klatki do wyliczenia opoznienia (latency)
            timestamp = time.perf_counter()
            frames_captured += 1

            if self.use_buffer:
                self.ring_buffer.put((frame, timestamp))
            else:
                with self.latest_lock:
                    # jesli poprzednia klatka nie zostala jeszcze odebrana przez main, to ja zgubilismy
                    if self.latest_frame is not None:
                        self.unbuffered_drops += 1
                    self.latest_frame = (frame, timestamp)

            # co 1 sekunde aktualizujemy realny FPS kamery
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time >= 1.0:
                self.capture_fps = frames_captured / elapsed_time
                frames_captured = 0
                start_time = time.perf_counter()

    def getFrame(self):
        if self.use_buffer:
            return self.ring_buffer.get()
        with self.latest_lock:
            item = self.latest_frame
            self.latest_frame = None
            return item

    def getDropStats(self) -> tuple[int, int, int]:
        if self.use_buffer:
            return self.ring_buffer.getStatsAndResetDrops()
        with self.latest_lock:
            drops = self.unbuffered_drops
            self.unbuffered_drops = 0
            buffer_fill = 1 if self.latest_frame is not None else 0
            return drops, buffer_fill, 1

    def stop(self) -> None:
        self.is_running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        self.capture.release()


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

    # bierzemy modul, zeby wykrywac zarowno jasne grzbiety na ciemnym tle, jak i ciemne obrecze na jasnym
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
    # rozmycie przed HoughCircles zabija szum matrycy, ktory tworzylby falszywe okregi
    blurred_image = cv.GaussianBlur(gray_image, (9, 9), 2)
    ridge_map = computeHessianRidges(gray_image, ridge_sigma=ridge_sigma)

    circles = cv.HoughCircles(
        blurred_image,      # obraz w skali szarosci (uint8)
        cv.HOUGH_GRADIENT,  # metoda gradientowa: wektor krawedzi wskazuje w strone srodka kola
        dp=1.2,             # odwrotnosc rozdzielczosci akumulatora (1.2 przyspiesza obliczenia)
        minDist=80,         # minimalna odleglosc miedzy srodkami wykrytych okregow
        param1=hough_param_1,  # gorny prog detektora krawedzi Canny'ego
        param2=hough_param_2,  # prog glosow w akumulatorze (im wyzszy, tym pewniejsze okregi)
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
    # mapa kolorow ulatwia wzrokowa ocene sily wykrytych grzbietow
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

    # puszczamy Transformate Hougha bezposrednio na mapie grzbietow (linii srodkowych obreczy)
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

    # przyciemniamy obraz z kamery i nakladamy na niebiesko wykryte grzbiety
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

# =====================================================================
# LAB 5: WYKRYWANIE CZERWONEJ KULKI (DEEP LEARNING - YOLO)
# =====================================================================

def loadYoloModel() -> YOLO | None:
    weights_path = Path(__file__).resolve().parent / "lab05" / "runs" / "red_ball_model" / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"[Lab 5] Nie znaleziono wag modelu: {weights_path}")
        return None
    return YOLO(str(weights_path))

# ladujemy wagi wytrenowanej sieci raz przy starcie programu
YOLO_MODEL = loadYoloModel()

def detectRedBallYolo(source: str | Path | np.ndarray, confidence_threshold: float = 0.50) -> np.ndarray | None:
    image = readImage(source)

    if image is None:
        return None

    if YOLO_MODEL is None:
        cv.putText(image, "Lab 5: Brak pliku best.pt!", (20, 200), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return image

    # verbose=False wylacza spamowanie konsoli logami przy kazdej pojedynczej klatce kamery
    prediction_results = YOLO_MODEL.predict(image, conf=confidence_threshold, imgsz=416, verbose=False)

    for result in prediction_results:
        for detected_box in result.boxes:
            # wyciagamy wspolrzedne prostokata (x1, y1, x2, y2) z tensora GPU do tablicy NumPy
            box_coordinates = detected_box.xyxy[0].cpu().numpy()
            top_left_x = int(box_coordinates[0])
            top_left_y = int(box_coordinates[1])
            bottom_right_x = int(box_coordinates[2])
            bottom_right_y = int(box_coordinates[3])

            confidence = float(detected_box.conf[0])
            label_text = f"YOLO red_ball ({confidence * 100:.1f}%)"

            drawBoundingBox(image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, label_text)

    return image

# =====================================================================
# GLOWNA PETLA KAMERY (LACZY LAB 1, LAB 2, LAB 3 I LAB 4)
# =====================================================================

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


def runCamera(window_title: str = "CvLabs (Lab 1 - Lab 4)", start_callback_index: int = 0) -> None:
    grabber = CameraGrabber(camera_source=0, buffer_capacity=30, use_buffer=True).start()

    # slownik z parametrami Lab 4 pozwala modyfikowac je z klawiatury w trakcie dzialania programu
    lab4_settings = {
        "hough_param_1": 100,
        "hough_param_2": 38,
        "ridge_sigma": 2.0,
    }

    # kazdy callback przyjmuje 1 argument (frame) i zwraca przetworzony obraz (np.ndarray)
    callbacks = [
        ("Lab 5: YOLO (Deep Learning)", detectRedBallYolo),
        (
            "Lab 4A: Hough (Kulka vs Obrecz)",
            lambda frame: detectCirclesHough(
                frame,
                hough_param_1=lab4_settings["hough_param_1"],
                hough_param_2=lab4_settings["hough_param_2"],
                ridge_sigma=lab4_settings["ridge_sigma"],
            ),
        ),
        (
            "Lab 4B: Detekcja Grzbietow (Hessian)",
            lambda frame: detectRidgesOnly(frame, ridge_sigma=lab4_settings["ridge_sigma"]),
        ),
        (
            "Lab 4C: Fuzja Grzbiety + Hough",
            lambda frame: detectRidgesAndHough(
                frame,
                hough_param_1=lab4_settings["hough_param_1"],
                hough_param_2=lab4_settings["hough_param_2"],
                ridge_sigma=lab4_settings["ridge_sigma"],
            ),
        ),
        ("Lab 1: Naiwna petla po pikselach", detectRedBallNaive),
        ("Lab 2: SimpleBlobDetector", detectRedBallBlob),
        ("Lab 3: Skala szarosci", convertToGrayscale),
        ("Brak (surowy obraz)", lambda frame: frame),
    ]

    callback_index = start_callback_index % len(callbacks)
    artificial_delay_seconds = 0.0

    displayed_frames = 0
    display_fps = 0.0
    dropped_fps = 0.0
    total_dropped = 0
    metrics_start_time = time.perf_counter()

    print("Sterowanie:")
    print("  [c]   - Zmiana algorytmu (Lab 4A -> 4B -> 4C -> Lab 1 -> Lab 2 -> Lab 3 -> Brak)")
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
            f"Lab 4: Hough param2 [w/s]={lab4_settings['hough_param_2']} | Sigma [e/d]={lab4_settings['ridge_sigma']:.1f}",
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
            lab4_settings["hough_param_2"] = min(120, lab4_settings["hough_param_2"] + 5)
        elif pressed_key == ord("s"):
            lab4_settings["hough_param_2"] = max(15, lab4_settings["hough_param_2"] - 5)
        elif pressed_key == ord("e"):
            lab4_settings["ridge_sigma"] = min(6.0, lab4_settings["ridge_sigma"] + 0.5)
        elif pressed_key == ord("d"):
            lab4_settings["ridge_sigma"] = max(1.0, lab4_settings["ridge_sigma"] - 0.5)

    grabber.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # test_image_path = Path("redBall.jpg")
    # showImageResult("Lab 1 - Naive", detectRedBallNaive(test_image_path))
    # showImageResult("Lab 2 - Blob", detectRedBallBlob(test_image_path))
    # showImageResult("Lab 4 - Hough", detectCirclesHough(test_image_path))

    runCamera()