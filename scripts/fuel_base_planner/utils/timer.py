import threading
import time

class MyTimer:
    def __init__(self, obj, interval_sec, callback):
        self.obj = obj
        self.interval = interval_sec
        self.callback = callback
        self.thread = threading.Thread(target=self._run)
        self.stop_event = threading.Event()

    def start(self):
        self.thread.start()

    def _run(self):
        next_time = time.time()
        while not self.stop_event.is_set():
            now = time.time()
            if now >= next_time:
                self.callback(self.obj)
                next_time += self.interval
            time.sleep(0.001)  # sleep a bit to avoid busy wait

    def stop(self):
        self.stop_event.set()
        self.thread.join()
