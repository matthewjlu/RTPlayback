import sys
import time
import threading
from queue import Queue, Empty

import numpy as np
import librosa
import pyrubberband as pyrb
import pyaudio
from PyQt5 import QtWidgets, QtCore

# ------------------------------
# Configuration
# ------------------------------
AUDIO_PATH = '/Users/mattlu/Desktop/test.wav'  # Replace with your file path
BATCH_DURATION = 1.0       # seconds per processing batch
PLAYBACK_CHUNK = 1024      # frames per playback write
SAMPLE_RATE = 44100
MIN_SPEED, MAX_SPEED = 0.1, 2.0
INACTIVITY_TIMEOUT = 3.0   # seconds without scroll before pausing

# ------------------------------
# Global State
# ------------------------------
stop_event = threading.Event()
audio_queue = Queue(maxsize=16)  # processed audio chunks
paused = threading.Event()       # when set, playback outputs silence
current_speed = 1.0              # playback speed multiplier

# To track when the last scroll event occurred
last_scroll_time_lock = threading.Lock()
last_scroll_time = time.time()

def clear_audio_queue():
    """Flush any pending audio chunks from the queue."""
    while True:
        try:
            audio_queue.get_nowait()
        except Empty:
            break

# ------------------------------
# Audio Processing Thread
# ------------------------------
def load_and_process_audio():
    global current_speed
    # Load and normalize the entire audio file.
    y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    total_samples = len(y)
    position = 0
    BATCH_SIZE = int(BATCH_DURATION * SAMPLE_RATE)

    while position < total_samples and not stop_event.is_set():
        if paused.is_set():
            # While paused, simply wait (do not advance the file pointer)
            time.sleep(0.1)
            continue

        end = min(position + BATCH_SIZE, total_samples)
        raw_chunk = y[position:end]
        position = end

        # Apply time-stretching if speed is not normal
        if current_speed != 1.0:
            try:
                stretched = pyrb.time_stretch(raw_chunk, sr, current_speed)
            except Exception as e:
                print("Time-stretch error:", e)
                stretched = raw_chunk
        else:
            stretched = raw_chunk

        # Split the processed batch into smaller chunks for playback.
        idx = 0
        while idx < len(stretched) and not stop_event.is_set():
            sub_end = min(idx + PLAYBACK_CHUNK, len(stretched))
            sub_chunk = stretched[idx:sub_end]
            idx = sub_end
            audio_queue.put(sub_chunk)

    # Signal the end of audio
    audio_queue.put(None)

# ------------------------------
# Audio Playback Thread
# ------------------------------
def audio_playback_worker():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=PLAYBACK_CHUNK
    )
    while not stop_event.is_set():
        if paused.is_set():
            # When paused, flush the audio queue and output silence.
            clear_audio_queue()
            silence = np.zeros(PLAYBACK_CHUNK, dtype=np.float32).tobytes()
            stream.write(silence)
            time.sleep(0.01)
            continue

        try:
            chunk = audio_queue.get(timeout=0.1)
        except Empty:
            continue
        if chunk is None:
            break
        data = np.array(chunk, dtype=np.float32).tobytes()
        stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

# ------------------------------
# PyQt5 GUI with Scroll Event
# ------------------------------
class ScrollControlWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audiobook Playback Speed Control")
        self.resize(300, 150)
        self.label = QtWidgets.QLabel("Speed: 1.00x", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # QTimer to monitor inactivity (no scroll events)
        self.inactivity_timer = QtCore.QTimer(self)
        self.inactivity_timer.setInterval(200)  # check every 200 ms
        self.inactivity_timer.timeout.connect(self.check_inactivity)
        self.inactivity_timer.start()

    def wheelEvent(self, event):
        global current_speed, paused, last_scroll_time
        delta = event.angleDelta().y() / 120  # typically ±1 per notch

        # Update last scroll time (thread-safe)
        with last_scroll_time_lock:
            last_scroll_time = time.time()

        if paused.is_set():
            # If previously paused, immediately resume playback.
            print("Resuming playback due to scroll input.")
            paused.clear()
            clear_audio_queue()
            # Set current_speed to the new target immediately.
            current_speed = max(MIN_SPEED, min(current_speed + delta, MAX_SPEED))
        else:
            current_speed = max(MIN_SPEED, min(current_speed + delta, MAX_SPEED))
        self.label.setText(f"Speed: {current_speed:.2f}x")
        print(f"Speed set to {current_speed:.2f}x")

    def check_inactivity(self):
        global last_scroll_time, paused
        with last_scroll_time_lock:
            elapsed = time.time() - last_scroll_time
        if elapsed > INACTIVITY_TIMEOUT and not paused.is_set():
            print("Inactivity detected. Pausing playback.")
            paused.set()
            clear_audio_queue()

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Create and show the PyQt5 application.
    app = QtWidgets.QApplication(sys.argv)
    window = ScrollControlWidget()
    window.show()

    # Start the audio processing and playback threads.
    processing_thread = threading.Thread(target=load_and_process_audio, daemon=True)
    playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
    processing_thread.start()
    playback_thread.start()

    ret = app.exec_()
    stop_event.set()
    processing_thread.join()
    playback_thread.join()
    sys.exit(ret)

if __name__ == "__main__":
    main()
