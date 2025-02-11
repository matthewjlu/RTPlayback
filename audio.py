import sys
import time
import numpy as np
from queue import Queue
from threading import Event, Thread
from PyQt5 import QtWidgets, QtCore
import pyaudio
import pyrubberband as pyrb
import librosa

# ------------------------------
# Global Configuration & State
# ------------------------------
AUDIO_PATH = '/Users/mattlu/Desktop/test.wav'
BATCH_DURATION = 1.0      # seconds per time-stretch batch (shorter batches reduce latency)
PLAYBACK_CHUNK = 2048     # frames per write to PyAudio
SAMPLE_RATE = 44100
BUFFER_SIZE = 8
MIN_SPEED, MAX_SPEED = 0.1, 2.0

# Smoothing: lower values yield smoother, less abrupt speed changes.
SMOOTHING_ALPHA = 0.1

# Global flags and queues.
stop_event = Event()         # Signals termination to threads.
speed_queue = Queue()        # To send updated speed values to the audio thread.
audio_queue = Queue(maxsize=BUFFER_SIZE)  # Audio data is passed via this queue.
current_speed = 1.0          # Global state: current playback speed.

# ------------------------------
# Audio Processing Functions
# ------------------------------
def load_and_process_audio():
    """
    Loads the audio file, normalizes it, and processes it in batches.
    Each batch is time-stretched using the most recent speed value.
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
        # Update the local speed if new speed values have been queued.
        while not speed_queue.empty():
            local_speed = speed_queue.get()
            local_speed = max(MIN_SPEED, min(local_speed, MAX_SPEED))
        
        end = min(position + BATCH_SIZE, total_samples)
        raw_chunk = y[position:end]
        position = end

        # Apply time-stretching if needed.
        if local_speed != 1.0:
            try:
                stretched = pyrb.time_stretch(raw_chunk, sr, local_speed)
            except Exception as e:
                print("Error in time stretching:", e)
                stretched = raw_chunk
        else:
            stretched = raw_chunk

        # Split the processed batch into smaller playback chunks.
        idx = 0
        while idx < len(stretched) and not stop_event.is_set():
            sub_end = min(idx + PLAYBACK_CHUNK, len(stretched))
            sub_chunk = stretched[idx:sub_end]
            idx = sub_end
            audio_queue.put(sub_chunk)
    
    # Signal end-of-audio.
    audio_queue.put(None)


def audio_playback_worker():
    """
    Continuously fetches audio chunks from the queue and writes them to the audio stream.
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
        try:
            stream.write(data)
        except Exception as e:
            print("Error writing audio:", e)
    
    stream.stop_stream()
    stream.close()
    p.terminate()


# ------------------------------
# PyQt5 SpeedControlWindow
# ------------------------------
class SpeedControlWindow(QtWidgets.QWidget):
    """
    A PyQt5 widget that adjusts playback speed via mouse wheel scrolling.
    
    Immediate changes:
      - Every scroll event immediately adjusts the speed (using an exponential moving average)
        and records the resulting speed with a timestamp.
    
    Auto-update:
      - A QTimer fires every 3 seconds.
      - If no scroll event has occurred in the last 3 seconds, the callback:
          * If recent scroll events (from the past 10 seconds) exist, computes their average and updates the speed.
          * If no scroll events are recorded, gradually recovers the speed toward 1.0.
      - This auto-update occurs repeatedly every 3 seconds if inactivity continues.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Audio Speed Control (Scroll to Adjust)")
        self.resize(300, 100)

        # Variables to track target and smoothed speeds.
        self.target_speed = 1.0
        self.smoothed_speed = 1.0

        # Label to display current playback speed.
        self.label = QtWidgets.QLabel("Playback Speed: 1.00x", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Quit button.
        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # List to record (timestamp, speed) for each scroll event.
        self.scroll_events = []

        # Timer for auto-update: fires every 3 seconds.
        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.setInterval(3000)  # 3000 ms = 3 seconds
        self.auto_timer.timeout.connect(self.auto_update_speed)
        self.auto_timer.start()

    def wheelEvent(self, event):
        """
        Handles mouse wheel events:
          - Updates the target speed (each notch changes speed by 0.1).
          - Uses an exponential moving average to smooth the speed.
          - Immediately sends the new speed to the audio thread.
          - Records the new (smoothed) speed with its timestamp.
        """
        global current_speed

        # Normalize the delta (one notch is typically ±120, so divide by 120).
        delta = event.angleDelta().y() / 120.0
        now = time.time()

        # Immediate adjustment: update target speed by 0.1 per notch.
        self.target_speed += delta * 0.1
        self.target_speed = max(MIN_SPEED, min(self.target_speed, MAX_SPEED))

        # Apply exponential moving average for smoothing.
        self.smoothed_speed = (SMOOTHING_ALPHA * self.target_speed +
                               (1 - SMOOTHING_ALPHA) * self.smoothed_speed)
        current_speed = self.smoothed_speed
        speed_queue.put(current_speed)
        self.label.setText(f"Playback Speed: {current_speed:.2f}x")
        print(f"Immediate speed changed to {current_speed:.2f}x via scroll event")

        # Record the new speed along with the timestamp.
        self.scroll_events.append((now, current_speed))
        # Keep only events from the past 10 seconds.
        self.scroll_events = [(t, s) for (t, s) in self.scroll_events if now - t <= 10]

    def auto_update_speed(self):
        """
        Called every 3 seconds.
        If no scroll event has occurred in the last 3 seconds, then:
          - If recent scroll events exist (within the past 10 seconds), compute their average and update the speed.
          - Otherwise, gradually recover the speed toward 1.0.
        """
        global current_speed
        now = time.time()

        # Determine the time of the most recent scroll event (if any).
        if self.scroll_events:
            most_recent = max(t for t, s in self.scroll_events)
        else:
            most_recent = 0

        if now - most_recent < 3:
            # The user has scrolled within the last 3 seconds; do not auto-update.
            return

        if self.scroll_events:
            # Compute average of speeds from the past 10 seconds.
            recent_events = [(t, s) for (t, s) in self.scroll_events if now - t <= 10]
            if recent_events:
                avg_speed = sum(s for (t, s) in recent_events) / len(recent_events)
            else:
                avg_speed = self.smoothed_speed
            # Update using the computed average.
            new_speed = avg_speed
            # Clear events to start fresh for the next period.
            self.scroll_events.clear()
        else:
            # No scroll events have been recorded in the last 10 seconds.
            # Gradually recover the speed toward 1.0.
            recovery_rate = 0.1  # Adjust recovery speed as needed.
            new_speed = self.smoothed_speed + (1.0 - self.smoothed_speed) * recovery_rate

        self.target_speed = new_speed
        self.smoothed_speed = new_speed
        current_speed = new_speed
        speed_queue.put(current_speed)
        self.label.setText(f"Playback Speed: {current_speed:.2f}x (Auto-updated)")
        print(f"Auto-updated speed to {current_speed:.2f}x")

    def close_app(self):
        """Stops audio processing and closes the application."""
        stop_event.set()
        self.close()


# ------------------------------
# Main Function
# ------------------------------
def main():
    # Start the audio processing and playback threads.
    processing_thread = Thread(target=load_and_process_audio, daemon=True)
    playback_thread = Thread(target=audio_playback_worker, daemon=True)
    processing_thread.start()
    playback_thread.start()

    # Set up and run the PyQt application.
    app = QtWidgets.QApplication(sys.argv)
    window = SpeedControlWindow()
    window.show()
    app.exec_()

    processing_thread.join()
    playback_thread.join()
    print("Playback stopped.")


if __name__ == "__main__":
    main()
