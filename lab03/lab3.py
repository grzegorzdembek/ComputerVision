import cv2
import threading

class lab3:
    class ThreadRingBuffer:
        def __init__(self, capacity: int):
            self.capacity = max(1, capacity) # rozmiar buffera > 0
            self.buffer = [None] * self.capacity # utworzenie pustego buffera
            self.head = 0  # odczyt (najstarsza ramka)
            self.tail = 0  # zapis (najnowsza ramka)
            self.count = 0 # ilosc ulokowanych ramek 
            self.dropped_frames = 0 # ilosc wyrzuconych ramek
            self.lock = threading.Lock() # 

    def put(self, item):
        with self.lock:
            if self.count == self.capacity: # pelny buffer
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