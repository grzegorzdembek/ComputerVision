import cv2
import numpy as np
import threading
import time

class ThreadSafeCircularBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self.buffer = [None] * self.capacity
        self.head = 0  
        self.tail = 0  
        self.count = 0
        self.dropped_frames = 0
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if self.count == self.capacity:
                self.head = (self.head + 1) % self.capacity
                self.dropped_frames += 1
            else:
                self.count += 1

            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity

    def get(self):
        with self.lock:
            if self.count == 0:
                return None
            item = self.buffer[self.head]
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
            return item

    def get_stats_and_reset_drops(self):
        with self.lock:
            drops = self.dropped_frames
            self.dropped_frames = 0
            return drops, self.count, self.capacity

class CameraGrabber:
    def __init__(self, src=0, buffer_capacity=30, use_buffer=True):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
       
        self.use_buffer = use_buffer
        self.ring_buffer = ThreadSafeCircularBuffer(buffer_capacity)

        self.latest_frame = None
        self.latest_lock = threading.Lock()
        self.unbuffered_drops = 0

        self.running = False
        self.thread = None
        self.capture_fps = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        frames_captured = 0
        t_start = time.perf_counter()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            timestamp = time.perf_counter()
            frames_captured += 1

            if self.use_buffer:
                self.ring_buffer.put((frame, timestamp))
            else:
                with self.latest_lock:
                    if self.latest_frame is not None:
                        self.unbuffered_drops += 1
                    self.latest_frame = (frame, timestamp)

            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                self.capture_fps = frames_captured / elapsed
                frames_captured = 0
                t_start = time.perf_counter()

    def get_frame(self):
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


def callback_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByColor = True
blob_params.blobColor = 0

blob_params.filterByArea = True
blob_params.minArea = 100
blob_params.maxArea = 5000000  

blob_params.filterByCircularity = True
blob_params.minCircularity = 0.80

blob_params.filterByConvexity = True
blob_params.minConvexity = 0.90

blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.85

detector_lab2 = cv2.SimpleBlobDetector_create(blob_params)

def callback_lab2(frame):
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([50, 50, 255])

    mask = cv2.inRange(frame, lower_red, upper_red)
    mask_inv = cv2.bitwise_not(mask)  

    keypoints = detector_lab2.detect(mask_inv)

    if len(keypoints) > 0:
        return cv2.drawKeypoints(
            frame,
            keypoints,
            np.array([]),
            (0, 255, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
    return frame

def main():
    grabber = CameraGrabber(src=0, buffer_capacity=30, use_buffer=True).start()

    callbacks = [
        ("Skala szarosci", callback_grayscale),
        ("Lab 2", callback_lab2),
        ("Brak ", lambda f: f),
    ]
    cb_idx = 1  

    artificial_delay_s = 0.0

    displayed_frames = 0
    display_fps = 0.0
    dropped_fps = 0.0
    total_dropped = 0
    t_metrics = time.perf_counter()

    print("Sterowanie:")
    print("  [b] - Bufor cykliczny (30 ramek) <-> Bez bufora")
    print("  [c] - Skala szarosci <-> Lab 2 <-> Brak)")
    print("  [+] - Zwieksz opoznienie (+10 ms)")
    print("  [-] - Zmniejsz opoznienie (-10 ms)")
    print("  [q] - Wyjscie")

    while True:
        item = grabber.get_frame()
        if item is None:
            time.sleep(0.001)
            continue

        frame, capture_timestamp = item

        cb_name, cb_func = callbacks[cb_idx]
        processed_frame = cb_func(frame)

        if artificial_delay_s > 0:
            time.sleep(artificial_delay_s)

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

        mode_str = f"Bufor cykliczny ({buf_fill}/{buf_cap})" if grabber.use_buffer else "Bez bufora"
        info_lines = [
            f"Tryb [b]: {mode_str}",
            f"Callback [c]: {cb_name} | Dodatkowe obciazenie [+/-]: {int(artificial_delay_s * 1000)} ms",
            f"Rozdzielczosc: {w}x{h} px",
            f"Kamera FPS: {grabber.capture_fps:.1f} | Wyswietlane FPS: {display_fps:.1f}",
            f"Gubione FPS: {dropped_fps:.1f}/s (Lacznie zgubiono: {total_dropped})",
            f"Opoznienie: {latency_ms:.1f} ms",
        ]

        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (5, 5), (600, 165), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, processed_frame, 0.4, 0, processed_frame)

        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if latency_ms < 100 else (0, 165, 255) if latency_ms < 300 else (0, 0, 255)
            cv2.putText(
                processed_frame, line, (15, 28 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA
            )

        cv2.imshow("Lab 3", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            grabber.use_buffer = not grabber.use_buffer
        elif key == ord('c'):
            cb_idx = (cb_idx + 1) % len(callbacks)
        elif key in (ord('+'), ord('=')):
            artificial_delay_s = min(0.5, artificial_delay_s + 0.01)
        elif key == ord('-'):
            artificial_delay_s = max(0.0, artificial_delay_s - 0.01)

    grabber.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()