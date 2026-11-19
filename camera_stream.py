import threading
import time

import cv2 as cv


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