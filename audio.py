import sys
from PyQt5 import QtWidgets, QtCore
import pyaudio
import pyrubberband as pyrb
import numpy as np
import librosa
from queue import Queue
from threading import Event, Thread
import time

# ------------------------------
# Configuration!
# ------------------------------
AUDIO_PATH = '/Users/mattlu/Desktop/test.wav'
BATCH_DURATION = 1.0   # seconds per time-stretch batch
PLAYBACK_CHUNK = 1024  # frames per write to PyAudio
SAMPLE_RATE = 44100
BUFFER_SIZE = 8
MIN_SPEED, MAX_SPEED = 0.1, 2.0

# ------------------------------
# Global state
# ------------------------------
stop_event = Event()
speed_queue = Queue()
audio_queue = Queue(maxsize=BUFFER_SIZE)
current_speed = 1.0      # Last speed value used by the audio thread

# Exponential moving average parameters
SMOOTHING_ALPHA = 0.2    # Lower values yield a smoother, less responsive average


def load_and_process_audio():
    """
    Loads and normalizes the audio file, then processes it in batches.
    Each batch is time-stretched according to the latest speed value received.
    """
    y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    total_samples = len(y)
    position = 0
    BATCH_SIZE = int(BATCH_DURATION * SAMPLE_RATE)
    local_speed = 1.0

    while position < total_samples and not stop_event.is_set():
        # Update local speed if any new value is in the queue.
        while not speed_queue.empty():
            local_speed = speed_queue.get()
            local_speed = max(MIN_SPEED, min(local_speed, MAX_SPEED))
        
        end = min(position + BATCH_SIZE, total_samples)
        raw_chunk = y[position:end]
        position = end

        # Apply time-stretching if needed.
        if local_speed != 1.0:
            stretched = pyrb.time_stretch(raw_chunk, sr, local_speed)
        else:
            stretched = raw_chunk

        # Split the processed batch into smaller playback chunks.
        idx = 0
        while idx < len(stretched) and not stop_event.is_set():
            sub_end = min(idx + PLAYBACK_CHUNK, len(stretched))
            sub_chunk = stretched[idx:sub_end]
            idx = sub_end
            audio_queue.put(sub_chunk)
    
    # Signal end-of-audio to the playback thread.
    audio_queue.put(None)


def audio_playback_worker():
    """
    Continuously fetches audio chunks from the queue and plays them using PyAudio.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=PLAYBACK_CHUNK
    )

    while not stop_event.is_set():
        chunk = audio_queue.get()
        if chunk is None:
            break
        data = chunk.astype(np.float32).tobytes()
        stream.write(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()


class SpeedControlWindow(QtWidgets.QWidget):
    """
    A PyQt5 widget that adjusts playback speed via mouse wheel scrolling.
    It applies an exponential moving average to smooth rapid changes.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Audio Speed Control (Scroll to Adjust)")
        self.resize(300, 100)

        # Variables to hold the target and smoothed speed.
        self.target_speed = 1.0
        self.smoothed_speed = 1.0

        # Label to display the current playback speed.
        self.label = QtWidgets.QLabel("Playback Speed: 1.00x", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        
        # Quit button.
        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

    def wheelEvent(self, event):
        """
        Handles mouse wheel events. Scrolling up increases the target speed,
        scrolling down decreases it. The new speed is smoothed using an exponential
        moving average, then sent to the audio processing thread.
        """
        global current_speed
        # event.angleDelta().y() returns multiples of 120 (one notch)
        delta = event.angleDelta().y() / 120  # typically ±1 per notch
        # Adjust target speed: each notch changes speed by 0.1.
        self.target_speed += delta * 0.1
        
        # Clamp the target speed between MIN_SPEED and MAX_SPEED.
        self.target_speed = max(MIN_SPEED, min(self.target_speed, MAX_SPEED))

        # Apply exponential moving average:
        self.smoothed_speed = SMOOTHING_ALPHA * self.target_speed + (1 - SMOOTHING_ALPHA) * self.smoothed_speed

        if self.smoothed_speed != current_speed:
            current_speed = self.smoothed_speed
            speed_queue.put(current_speed)
            self.label.setText(f"Playback Speed: {current_speed:.2f}x")
            print(f"Speed changed to {current_speed:.2f}x")

    def close_app(self):
        """Stops audio processing and closes the application."""
        stop_event.set()
        self.close()


def main():
    # Start the audio processing and playback threads.
    processing_thread = Thread(target=load_and_process_audio, daemon=True)
    playback_thread = Thread(target=audio_playback_worker, daemon=True)
    processing_thread.start()
    playback_thread.start()

    # Set up the PyQt5 application and display the window.
    app = QtWidgets.QApplication(sys.argv)
    window = SpeedControlWindow()
    window.show()
    app.exec_()

    # Wait for audio threads to finish after closing the window.
    processing_thread.join()
    playback_thread.join()
    print("Playback stopped")

if __name__ == "__main__":
    main()
