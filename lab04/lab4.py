import cv2
import numpy as np
import threading
import time


# =====================================================================
# CZĘŚĆ 1: ARCHITEKTURA Z LAB 3 (BUFOR CYKLICZNY I WĄTEK KAMERY)
# =====================================================================

class ThreadSafeCircularBuffer:
    """
    Bezpieczny wątkowo bufor cykliczny (Ring Buffer) o stałej pojemności.
    Działa jak taśmociąg zamknięty w pętli:
    - Wątek kamery (Producent) wkłada klatki pod indeks 'tail' (ogon).
    - Wątek główny (Konsument) wyjmuje klatki spod indeksu 'head' (głowa).
    """
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        # Inicjalizujemy pustą listę o stałym rozmiarze
        self.buffer = [None] * self.capacity
        self.head = 0          # Indeks najstarszego elementu (do odczytu)
        self.tail = 0          # Indeks wolnego miejsca (do zapisu)
        self.count = 0         # Aktualna liczba klatek w buforze
        self.dropped_frames = 0 # Licznik zgubionych (nadpisanych) klatek
        # Zamek (Mutex) gwarantuje, że dwa wątki nie zmodyfikują bufora w tym samym ułamku sekundy
        self.lock = threading.Lock()

    def put(self, item):
        """Dodaje nową klatkę do bufora (wywoływane przez wątek kamery)."""
        with self.lock: # Blokujemy dostęp dla innych wątków na czas zapisu
            if self.count == self.capacity:
                # Bufor jest pełny! Nadpisujemy najstarszą klatkę, więc przesuwamy 'head'
                # Operacja modulo (%) sprawia, że po indeksie 29 wracamy na indeks 0 (cykliczność)
                self.head = (self.head + 1) % self.capacity
                self.dropped_frames += 1
            else:
                self.count += 1

            # Zapisujemy klatkę i przesuwamy wskaźnik zapisu 'tail'
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity

    def get(self):
        """Pobiera najstarszą klatkę z bufora (wywoływane przez pętlę główną)."""
        with self.lock:
            if self.count == 0:
                return None # Bufor pusty - kamera nie zdążyła jeszcze dać nowej klatki
            
            item = self.buffer[self.head]
            self.buffer[self.head] = None # Czyścimy referencję dla Garbage Collectora
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
            return item

    def get_stats_and_reset_drops(self):
        """Zwraca statystyki bufora i zeruje licznik zgubionych klatek dla kolejnego pomiaru."""
        with self.lock:
            drops = self.dropped_frames
            self.dropped_frames = 0
            return drops, self.count, self.capacity


class CameraGrabber:
    """
    Klasa obsługująca kamerę w osobnym wątku w tle (nieblokujące I/O).
    Dzięki niej powolne przetwarzanie obrazu (np. Hough lub pętla z Lab 1)
    nie blokuje samego procesu odbierania klatek ze sterownika kamery.
    """
    def __init__(self, src=0, width=1280, height=720, buffer_capacity=30, use_buffer=True):
        self.cap = cv2.VideoCapture(src)
        # Ustawiamy rozdzielczość (1280x720 lub 1920x1080 w zależności od kamery)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Zmniejszamy wewnętrzny bufor sterownika OpenCV do 1 klatki, aby uniknąć ukrytych opóźnień
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.use_buffer = use_buffer
        self.ring_buffer = ThreadSafeCircularBuffer(buffer_capacity)

        # Zmienne dla trybu "Bez bufora" (zawsze bierzemy tylko najświeższą klatkę)
        self.latest_frame = None
        self.latest_lock = threading.Lock()
        self.unbuffered_drops = 0

        self.running = False
        self.thread = None
        self.capture_fps = 0.0

    def start(self):
        """Uruchamia wątek zbierający klatki w tle (daemon=True zamknie wątek wraz z programem)."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        """Pętla działająca w tle - odbiera klatki tak szybko, jak pozwala kamera."""
        frames_captured = 0
        t_start = time.perf_counter()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Zapisujemy dokładny czas pobrania klatki, by policzyć opóźnienie (latency)
            timestamp = time.perf_counter()
            frames_captured += 1

            if self.use_buffer:
                self.ring_buffer.put((frame, timestamp))
            else:
                with self.latest_lock:
                    # Jeśli poprzednia klatka nie została jeszcze odebrana przez main(), to znaczy że ją zgubiliśmy
                    if self.latest_frame is not None:
                        self.unbuffered_drops += 1
                    self.latest_frame = (frame, timestamp)

            # Co 1 sekundę aktualizujemy realny FPS kamery
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self.capture_fps = frames_captured / elapsed
                frames_captured = 0
                t_start = time.perf_counter()

    def get_frame(self):
        """Pobiera klatkę (z bufora cyklicznego lub bezpośrednio najnowszą)."""
        if self.use_buffer:
            return self.ring_buffer.get()
        else:
            with self.latest_lock:
                item = self.latest_frame
                self.latest_frame = None
                return item

    def get_drop_stats(self):
        if self.use_buffer:
            return self.ring_buffer.get_stats_and_reset_drops()
        else:
            with self.latest_lock:
                drops = self.unbuffered_drops
                self.unbuffered_drops = 0
                fill = 1 if self.latest_frame is not None else 0
                return drops, fill, 1

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.cap.release()


# =====================================================================
# CZĘŚĆ 2: WCZEŚNIEJSZE LABORATORIA PODPIĘTE POD KAMERĘ (LAB 1 i LAB 2)
# =====================================================================

def callback_lab1(frame):
    """
    Adaptacja Lab 1 (ręczne przeszukiwanie pikseli + równanie koła) pod kamerę na żywo.
    UWAGA EDUKACYJNA: W czystym Pythonie podwójna pętla 'for' po obrazie Full HD (2 miliony pikseli)
    trwa ~3 sekundy na jedną klatkę! Dlatego zmniejszamy obraz roboczy do szerokości 160 px,
    wykonujemy dokładnie Twój algorytm z Lab 1, a wynik przeskalowujemy z powrotem na duży ekran.
    """
    output = frame.copy()
    orig_h, orig_w = frame.shape[:2]

    # Skalujemy obraz w dół (np. 8-krotnie), żeby pętla 'for' działała w czasie rzeczywistym
    scale = 160.0 / orig_w
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    height, width, _ = small_frame.shape

    min_x, max_x = width, 0
    min_y, max_y = height, 0
    red_pixels = []

    # Dokładnie Twoja pętla z Lab 1: badanie składowych B, G, R każdego piksela
    for y in range(height):
        for x in range(width):
            b, g, r = small_frame[y, x]
            if int(r) > 140 and int(g) < 60 and int(b) < 60:
                red_pixels.append((x, y))
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y

    pixel_count = len(red_pixels)
    if pixel_count < 15:
        cv2.putText(output, "Lab 1: Brak czerwonych pikseli", (20, orig_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output

    # Wymiary prostokąta otaczającego (Bounding Box)
    bb_width = max_x - min_x + 1
    bb_height = max_y - min_y + 1
    bb_area = bb_width * bb_height

    # Stosunek wypełnienia: koło wpisane w kwadrat zajmuje pi * r^2 / (2r)^2 = pi / 4 ~= 0.785
    fill_ratio = pixel_count / bb_area
    is_symmetric = 0.75 <= (bb_width / bb_height) <= 1.25

    # Środek i promień hipotetycznej kulki
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    radius = (bb_width + bb_height) / 4.0

    # Test równania koła: (x - cx)^2 + (y - cy)^2 <= r^2
    # Dodajemy lekki margines 1.05 na dyskretyzację pikseli z kamery
    pixels_inside_circle = 0
    r_sq = (radius * 1.05) ** 2
    for x, y in red_pixels:
        if (x - center_x)**2 + (y - center_y)**2 <= r_sq:
            pixels_inside_circle += 1

    inside_ratio = pixels_inside_circle / pixel_count

    # Przeliczamy współrzędne z małego obrazka z powrotem na dużą klatkę kamery
    inv_scale = 1.0 / scale
    rx1, ry1 = int(min_x * inv_scale), int(min_y * inv_scale)
    rx2, ry2 = int(max_x * inv_scale), int(max_y * inv_scale)
    rcx, rcy = int(center_x * inv_scale), int(center_y * inv_scale)
    rr = int(radius * inv_scale)

    # Weryfikacja warunków z Lab 1
    if 0.60 <= fill_ratio <= 0.90 and is_symmetric and inside_ratio >= 0.92:
        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.circle(output, (rcx, rcy), rr, (0, 255, 0), 3)
        cv2.putText(output, f"Lab 1: Wykryto czerwona kulke! (fill={fill_ratio:.2f})",
                    (20, orig_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
        cv2.putText(output, f"Lab 1: Wykryto cos innego (fill={fill_ratio:.2f}, sym={is_symmetric})",
                    (20, orig_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    return output


# Konfiguracja detektora z Lab 2 (SimpleBlobDetector)
blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByColor = True
blob_params.blobColor = 0             # Szukamy czarnych plam na odwróconej masce
blob_params.filterByArea = True
blob_params.minArea = 150
blob_params.maxArea = 5000000
blob_params.filterByCircularity = True
blob_params.minCircularity = 0.75     # Okrągłość (1.0 to idealne koło)
blob_params.filterByConvexity = True
blob_params.minConvexity = 0.85       # Wypukłość
blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.75    # Bezwładność (wydłużenie elipsy)

detector_lab2 = cv2.SimpleBlobDetector_create(blob_params)

def callback_lab2(frame):
    """Callback z Lab 2: progowanie koloru czerwonego + SimpleBlobDetector."""
    lower_red = np.array([0, 0, 140])
    upper_red = np.array([65, 65, 255])

    mask = cv2.inRange(frame, lower_red, upper_red)
    # Oczyszczamy maskę z szumu kamery za pomocą operacji morfologicznego domknięcia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_inv = cv2.bitwise_not(mask)

    keypoints = detector_lab2.detect(mask_inv)
    output = frame.copy()
    if len(keypoints) > 0:
        output = cv2.drawKeypoints(
            output, keypoints, np.array([]), (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    return output


def callback_grayscale(frame):
    """Bazowy callback z Lab 3: konwersja do skali szarości."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# =====================================================================
# CZĘŚĆ 3: LAB 4 - TRANSFORMATA HOUGHA I WYKRYWANIE GRZBIETÓW (NOWOŚĆ!)
# =====================================================================

class Lab4Parameters:
    """
    Klasa przechowująca parametry dla Lab 4, które można regulować na żywo z klawiatury!
    """
    def __init__(self):
        # Parametry cv2.HoughCircles:
        # param1: wyższy próg dla wewnętrznego detektora krawędzi Canny'ego
        self.hough_param1 = 100
        # param2: próg akumulatora (im MNIEJSZY, tym więcej okręgów wykryje, ale też więcej fałszywych;
        #         im WIĘKSZY, tym bardziej idealny musi być okrąg)
        self.hough_param2 = 38
        # Grubość poszukiwanego grzbietu (odchylenie standardowe rozmycia Gaussa przed drugą pochodną)
        self.ridge_sigma = 2.0


def compute_hessian_ridges(gray_img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    PROPOZYCJA IMPLEMENTACJI WYKRYWANIA GRZBIETÓW (Ridge Detection) za pomocą Macierzy Hessego.
    
    Jak to działa krok po kroku?
    1. Rozmywamy obraz filtrem Gaussa o skali 'sigma'. Sigma decyduje o tym, jak szerokich
       grzbietów (obręczy) szukamy. Rozmycie usuwa też szum, który zepsułby drugie pochodne.
    2. Liczymy pierwsze pochodne kierunkowe (gradienty) w osi X oraz Y operatorem Sobela: Ix, Iy.
    3. Liczymy drugie pochodne (krzywiznę terenu) różniczkując ponownie:
       - Ixx (zmiana gradientu poziomego w poziomie)
       - Iyy (zmiana gradientu pionowego w pionie)
       - Ixy (pochodna mieszana - zmiana gradientu poziomego w pionie)
    4. Dla każdego piksela mamy macierz Hessego 2x2:
       H = [ Ixx  Ixy ]
           [ Ixy  Iyy ]
    5. Liczymy wartości własne (eigenvalues) tej macierzy ze wzoru analitycznego:
       lambda_1,2 = 0.5 * ( (Ixx + Iyy) +/- sqrt((Ixx - Iyy)^2 + 4 * Ixy^2) )
    6. Maksymalna co do modułu wartość własna mówi nam, jak silny jest grzbiet w danym punkcie,
       niezależnie od tego, pod jakim kątem biegnie linia obręczy!
    """
    # Krok 1: Konwersja na liczby zmiennoprzecinkowe (float32) i wygładzanie Gaussa
    img_float = gray_img.astype(np.float32) / 255.0
    # Rozmiar maski Gaussa wyznaczamy automatycznie z parametru sigma (zasada 6*sigma + 1)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    smoothed = cv2.GaussianBlur(img_float, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # Krok 2: Pierwsze pochodne cząstkowe (Sobel 3x3)
    Ix = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)

    # Krok 3: Drugie pochodne cząstkowe (elementy macierzy Hessego)
    Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)

    # Krok 4 i 5: Obliczenie wartości własnych macierzy Hessego dla wszystkich pikseli naraz (wektorowo!)
    # Ślad macierzy (Trace) = Ixx + Iyy
    trace = Ixx + Iyy
    # Pierwiastek z wyróżnika: sqrt((Ixx - Iyy)^2 + 4 * Ixy^2)
    diff = Ixx - Iyy
    discriminant = np.sqrt(diff * diff + 4.0 * (Ixy * Ixy))

    lambda1 = 0.5 * (trace + discriminant)
    lambda2 = 0.5 * (trace - discriminant)

    # Krok 6: Wybór największej krzywizny (zarówno dla jasnych grzbietów jak i ciemnych obręczy)
    # Jasny grzbiet na ciemnym tle ma silnie ujemną drugą pochodną (wklęsły szczyt góry: lambda < 0).
    # Ciemna obręcz na jasnym tle ma silnie dodatnią drugą pochodną (dno doliny: lambda > 0).
    # Biorąc maksimum z wartości bezwzględnych |lambda1| i |lambda2|, wykrywamy oba typy obręczy!
    ridge_strength = np.maximum(np.abs(lambda1), np.abs(lambda2))

    # Normalizacja wyniku do zakresu 0..255 (uint8), aby można go było wyświetlić i podać do Hougha
    ridge_norm = cv2.normalize(ridge_strength, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Progowanie eliminujące słaby szum tła
    _, ridge_binary = cv2.threshold(ridge_norm, 35, 255, cv2.THRESH_TOZERO)
    return ridge_binary


def classify_ball_vs_rim(gray_frame: np.ndarray, ridge_map: np.ndarray, cx: int, cy: int, r: int) -> str:
    """
    Rozróżnia, czy wykryty przez Transformatę Hougha okrąg (cx, cy, r) to:
    - PEŁNA KULKA (jednolity środek, brak pustej przestrzeni wewnątrz)
    - OBRĘCZ / PIERŚCIEŃ (silny grzbiet na obwodzie 'r' oraz wyraźnie inne/puste wnętrze)
    """
    h, w = gray_frame.shape[:2]
    # Tworzymy dwie maski kołowe:
    # 1. Maska samego środka obiektu (promień od 0 do 0.5 * r)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_inner, (cx, cy), max(2, int(r * 0.5)), 255, -1)

    # 2. Maska pierścienia na obwodzie (od 0.85 * r do 1.15 * r) - tam gdzie biegnie obręcz
    mask_ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_ring, (cx, cy), int(r * 1.10), 255, -1)
    cv2.circle(mask_ring, (cx, cy), int(r * 0.85), 0, -1)

    # Sprawdzamy dwa kryteria:
    # Kryterium A: Różnica jasności między samym środkiem a pierścieniem obręczy
    mean_inner = cv2.mean(gray_frame, mask=mask_inner)[0]
    mean_ring = cv2.mean(gray_frame, mask=mask_ring)[0]
    intensity_diff = abs(mean_inner - mean_ring)

    # Kryterium B: Czy wewnątrz obiektu (poza obwodem) jest gładko, czy widać wewnętrzną krawędź obręczy?
    # Dla obręczy z dziurą w środku strefa przejściowa (0.5*r .. 0.85*r) ma silny skok grzbietu/krawędzi
    mask_transition = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_transition, (cx, cy), int(r * 0.85), 255, -1)
    cv2.circle(mask_transition, (cx, cy), int(r * 0.45), 0, -1)
    transition_ridge = cv2.mean(ridge_map, mask=mask_transition)[0]

    # Jeśli środek ma zupełnie inną jasność niż brzeg LUB wewnątrz widać drugą ściankę pierścienia -> OBRĘCZ
    if intensity_diff > 22.0 or transition_ridge > 45.0:
        return "OBRECZ (Ring)"
    else:
        return "KULKA (Ball)"


def callback_lab4_hough(frame, params: Lab4Parameters):
    """
    LAB 4A: Główny callback wykorzystujący Transformatę Hougha (cv2.HoughCircles)
    do znajdowania kulki lub obręczy wraz z klasyfikacją obiektu.
    """
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # KROK 1: Wygładzanie obrazu przed Transformatą Hougha.
    # To absolutnie kluczowy krok! Bez rozmycia detektor krawędzi wewnątrz HoughCircles
    # wykryje tysiące drobnych szumów z matrycy kamery i stworzy fałszywe okręgi.
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Liczymy pomocniczą mapę grzbietów do klasyfikacji Kulka vs Obręcz
    ridge_map = compute_hessian_ridges(gray, sigma=params.ridge_sigma)

    # KROK 2: Wywołanie Transformacji Hougha dla okręgów (Metoda Gradientowa)
    # Argumenty funkcji cv2.HoughCircles:
    # - blurred: 8-bitowy obraz w skali szarości
    # - cv2.HOUGH_GRADIENT: metoda wykorzystująca gradienty Sobela, by głosować na środki (a, b)
    # - dp=1.2: odwrotność rozdzielczości akumulatora (1.2 oznacza akumulator nieco mniejszy niż obraz, co przyspiesza obliczenia)
    # - minDist=80: minimalna odległość w pikselach między środkami dwóch wykrytych okręgów (zapobiega duplikatom)
    # - param1: górny próg dla detektora krawędzi Canny'ego (dolny jest automatycznie 2x mniejszy)
    # - param2: próg głosów w akumulatorze (im wyższy, tym pewniejsze okręgi)
    # - minRadius, maxRadius: minimalny i maksymalny promień szukanego obiektu w pikselach
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=params.hough_param1,
        param2=params.hough_param2,
        minRadius=20,
        maxRadius=300
    )

    # KROK 3: Rysowanie i analiza znalezionych okręgów
    if circles is not None:
        # Zaokrąglamy współrzędne (x, y, r) do liczb całkowitych
        circles_rounded = np.uint16(np.around(circles))

        for circle in circles_rounded[0, :]:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])

            # Sprawdzamy, czy okrąg to pełna kulka, czy obręcz
            obj_type = classify_ball_vs_rim(gray, ridge_map, cx, cy, r)
            color = (0, 255, 0) if "KULKA" in obj_type else (255, 200, 0)

            # Rysujemy obrys wykrytego okręgu
            cv2.circle(output, (cx, cy), r, color, 3)
            # Rysujemy mały czerwony punkt w samym środku okręgu (a, b)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            # Podpisujemy wykryty obiekt i jego promień
            label = f"{obj_type} (r={r}px)"
            cv2.putText(output, label, (max(10, cx - r), max(30, cy - r - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    return output


def callback_lab4_ridges_only(frame, params: Lab4Parameters):
    """
    LAB 4B: Wizualizacja samego wykrywania grzbietów (Ridge Detection - Macierz Hessego).
    Pozwala pokazać prowadzącemu, czym różni się mapa grzbietów od zwykłego obrazu:
    cienkie obręcze, przewody czy kontury świecą tu jako wyraźne linie szczytowe.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ridges = compute_hessian_ridges(gray, sigma=params.ridge_sigma)

    # Nakładamy mapę kolorystyczną (COLORMAP_INFERNO lub JET), żeby siła grzbietu była czytelna
    ridges_colored = cv2.applyColorMap(ridges, cv2.COLORMAP_INFERNO)
    return ridges_colored


def callback_lab4_ridges_plus_hough(frame, params: Lab4Parameters):
    """
    LAB 4C: Fuzja Wykrywania Grzbietów z Transformatą Hougha!
    Zamiast podawać do Transformacji Hougha zwykły obraz w skali szarości,
    najpierw wyodrębniamy z obrazu GRZBIETY (linie środkowe obręczy za pomocą macierzy Hessego),
    a następnie szukamy okręgów bezpośrednio na mapie grzbietów!
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ridges = compute_hessian_ridges(gray, sigma=params.ridge_sigma)

    # Lekko wygładzamy mapę grzbietów, by gradienty w HoughCircles były ciągłe
    ridges_smooth = cv2.GaussianBlur(ridges, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        ridges_smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=params.hough_param1,
        param2=max(15, params.hough_param2 - 8),
        minRadius=20,
        maxRadius=300
    )

    # Tworzymy obraz wyjściowy: przyciemniony obraz z kamery + podświetlone na niebiesko grzbiety
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


# =====================================================================
# CZĘŚĆ 4: PĘTLA GŁÓWNA PROGRAMU I INTERFEJS UŻYTKOWNIKA (OSD)
# =====================================================================

def main():
    # Inicjalizacja wątku kamery z buforem cyklicznym na 30 ramek
    grabber = CameraGrabber(src=0, width=1280, height=720, buffer_capacity=30, use_buffer=True).start()
    lab4_params = Lab4Parameters()

    # Lista wszystkich dostępnych algorytmów (przełączanych klawiszem [c])
    callbacks = [
        ("Lab 4A: Hough (Kulka vs Obrecz)", lambda f: callback_lab4_hough(f, lab4_params)),
        ("Lab 4B: Detekcja Grzbietow (Hessian)", lambda f: callback_lab4_ridges_only(f, lab4_params)),
        ("Lab 4C: Fuzja Grzbiety + Hough", lambda f: callback_lab4_ridges_plus_hough(f, lab4_params)),
        ("Lab 1: Piksele + Rownanie kola", callback_lab1),
        ("Lab 2: SimpleBlobDetector", callback_lab2),
        ("Lab 3: Skala szarosci", callback_grayscale),
        ("Brak (Surowy obraz)", lambda f: f),
    ]
    cb_idx = 0 # Domyślnie startujemy od głównego zadania Lab 4A

    artificial_delay_s = 0.0
    displayed_frames = 0
    display_fps = 0.0
    dropped_fps = 0.0
    total_dropped = 0
    t_metrics = time.perf_counter()

    print("=================================================================")
    print(" STEROWANIE (aktywne okno obrazu):")
    print("  [c]   - Zmiana algorytmu (Lab 4A -> Lab 4B -> Lab 4C -> Lab 1 -> Lab 2 -> Lab 3)")
    print("  [b]   - Przelacz: Bufor cykliczny (30 ramek) <-> Bez bufora")
    print("  [w/s] - Prog akumulatora Hougha param2 (+5 / -5) [czulosc okregow]")
    print("  [e/d] - Skala grzbietow Sigma (+0.5 / -0.5) [grubosc szukanej obreczy]")
    print("  [+/-] - Sztuczne obciazenie procesora (+10 ms / -10 ms)")
    print("  [q]   - Wyjscie z programu")
    print("=================================================================")

    while True:
        item = grabber.get_frame()
        if item is None:
            time.sleep(0.001)
            continue

        frame, capture_timestamp = item

        # Wykonanie aktualnie wybranego algorytmu przetwarzania obrazu
        cb_name, cb_func = callbacks[cb_idx]
        processed_frame = cb_func(frame)

        # Symulacja dodatkowego obciążenia obliczeniowego (z Lab 3)
        if artificial_delay_s > 0:
            time.sleep(artificial_delay_s)

        # Pomiar czasu i opóźnienia (latency) od momentu pobrania klatki z kamery do teraz
        now = time.perf_counter()
        latency_ms = (now - capture_timestamp) * 1000.0
        h, w = processed_frame.shape[:2]

        displayed_frames += 1
        elapsed = now - t_metrics
        if elapsed >= 0.5:
            display_fps = displayed_frames / elapsed
            drops, buf_fill, buf_cap = grabber.get_drop_stats()
            dropped_fps = drops / elapsed
            total_dropped += drops
            displayed_frames = 0
            t_metrics = now
        else:
            buf_fill = grabber.ring_buffer.count if grabber.use_buffer else 0
            buf_cap = grabber.ring_buffer.capacity

        # Przygotowanie panelu informacyjnego na ekranie (HUD)
        mode_str = f"Bufor cykliczny ({buf_fill}/{buf_cap})" if grabber.use_buffer else "Bez bufora (Latest Frame)"
        info_lines = [
            f"Algorytm [c]: {cb_name}",
            f"Tryb bufora [b]: {mode_str} | Sztuczne opoznienie [+/-]: {int(artificial_delay_s * 1000)} ms",
            f"Parametry Lab4: Hough prog [w/s]={lab4_params.hough_param2} | Grzbiet sigma [e/d]={lab4_params.ridge_sigma:.1f}",
            f"Rozdzielczosc: {w}x{h} px | Opoznienie (Latency): {latency_ms:.1f} ms",
            f"Kamera FPS: {grabber.capture_fps:.1f} | Wyswietlane FPS: {display_fps:.1f} | Gubione: {dropped_fps:.1f}/s (Suma: {total_dropped})",
        ]

        # Półprzezroczyste czarne tło pod napisy
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (5, 5), (780, 145), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, processed_frame, 0.35, 0, processed_frame)

        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if latency_ms < 100 else (0, 165, 255) if latency_ms < 300 else (0, 0, 255)
            if i == 0:
                color = (0, 255, 255) # Wyróżnienie nazwy aktualnego algorytmu na żółto
            cv2.putText(
                processed_frame, line, (15, 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA
            )

        cv2.imshow("Computer Vision - Lab 4 (Hough & Grzbiety)", processed_frame)

        # Obsługa klawiatury
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            grabber.use_buffer = not grabber.use_buffer
        elif key == ord('c'):
            cb_idx = (cb_idx + 1) % len(callbacks)
        elif key == ord('w'):
            lab4_params.hough_param2 = min(120, lab4_params.hough_param2 + 5)
        elif key == ord('s'):
            lab4_params.hough_param2 = max(15, lab4_params.hough_param2 - 5)
        elif key == ord('e'):
            lab4_params.ridge_sigma = min(6.0, lab4_params.ridge_sigma + 0.5)
        elif key == ord('d'):
            lab4_params.ridge_sigma = max(1.0, lab4_params.ridge_sigma - 0.5)
        elif key in (ord('+'), ord('=')):
            artificial_delay_s = min(0.5, artificial_delay_s + 0.01)
        elif key == ord('-'):
            artificial_delay_s = max(0.0, artificial_delay_s - 0.01)

    grabber.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()