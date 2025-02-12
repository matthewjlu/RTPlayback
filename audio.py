import sys
import os
import time
import numpy as np
from queue import Queue, Empty
from threading import Event, Thread
from PyQt5 import QtWidgets, QtCore
import pyaudio
import pyrubberband as pyrb
import librosa

# ------------------------------
# Global Configuration & State
# ------------------------------
file_name = "test.wav"
AUDIO_PATH = os.path.abspath(file_name)
BATCH_DURATION = 1.0      # seconds per time-stretch batch
PLAYBACK_CHUNK = 2048     # frames per write to PyAudio
SAMPLE_RATE = 44100
BUFFER_SIZE = 8
MIN_SPEED, MAX_SPEED = 0.1, 2.0
TIME = 15  # seconds in the past used for averaging

SMOOTHING_ALPHA = 0.1  # for exponential smoothing

# Global flags and queues.
stop_event = Event()         # Signals termination to threads.
speed_queue = Queue()        # To send updated speed values to the audio thread.
audio_queue = Queue(maxsize=BUFFER_SIZE)  # Audio data is passed via this queue.
current_speed = 1.0          # Global state: current playback speed.
paused = False               # Global flag for whether playback is paused.

# Option: set to True to use Gaussian weighting instead of inverse-time weighting.
use_gaussian = True

# ------------------------------
# Gaussian Weighting Function
# ------------------------------
def gaussian_weighted_average(recent_events, now, mu=0, sigma=TIME/2.0, epsilon=1e-6):
    """
    Computes the weighted average of speeds from recent_events using Gaussian weights.
    
    Each event is a tuple (t, s), where t is the timestamp and s is the speed.
    The weight is computed as:
         weight = exp(-((now - t - mu)**2) / (2 * sigma**2))
    """
    weighted_sum = sum(s * np.exp(-((now - t - mu)**2) / (2 * sigma**2)) for (t, s) in recent_events)
    total_weight = sum(np.exp(-((now - t - mu)**2) / (2 * sigma**2)) for (t, s) in recent_events)
    return weighted_sum / (total_weight + epsilon)

# ------------------------------
# Audio Processing Functions
# ------------------------------
def load_and_process_audio():
    """
    Loads the audio file, normalizes it, and processes it in batches.
    Each batch is time-stretched using the most recent speed value.
    When the speed is near 0, outputs silence.
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
        while not speed_queue.empty():
            local_speed = speed_queue.get()
            local_speed = max(MIN_SPEED, min(local_speed, MAX_SPEED))
        
        end = min(position + BATCH_SIZE, total_samples)
        raw_chunk = y[position:end]
        position = end

        if local_speed < 0.07:
            stretched = np.zeros_like(raw_chunk)
        elif local_speed != 1.0:
            try:
                stretched = pyrb.time_stretch(raw_chunk, sr, local_speed)
            except Exception as e:
                print("Error in time stretching:", e)
                stretched = raw_chunk
        else:
            stretched = raw_chunk

        idx = 0
        while idx < len(stretched) and not stop_event.is_set():
            sub_end = min(idx + PLAYBACK_CHUNK, len(stretched))
            sub_chunk = stretched[idx:sub_end]
            idx = sub_end
            audio_queue.put(sub_chunk)
    
    audio_queue.put(None)

def audio_playback_worker():
    """
    Continuously fetches audio chunks from the queue and writes them to the audio stream.
    If playback is paused, the thread sleeps without writing audio.
    """
    global paused
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=PLAYBACK_CHUNK
    )

    while not stop_event.is_set():
        if paused:
            if stream.is_active():
                stream.stop_stream()
            time.sleep(0.1)
            continue
        else:
            if not stream.is_active():
                stream.start_stream()
        try:
            chunk = audio_queue.get(timeout=0.1)
        except Empty:
            continue
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
    
    • Immediate changes: each wheel event updates the speed and sends it to the audio thread.
    • Scroll-session recording: records only the start and end of each scroll session.
    • Auto-update: every 3 seconds, if no scroll occurs, computes a weighted average (using inverse‑time or Gaussian weighting)
      or decays the speed toward 0. If the computed speed is below 0.07, playback is paused.
    • A Pause/Resume button allows manual control.
    • Gradual speed updates (over a few steps) are used to smooth the transitions and avoid popping.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Audio Speed Control (Scroll to Adjust)")
        self.resize(300, 160)

        self.target_speed = 1.0
        self.smoothed_speed = 1.0

        self.label = QtWidgets.QLabel("Playback Speed: 1.00x", self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.pause_button = QtWidgets.QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        self.scroll_events = []
        self.scroll_session_active = False

        self.scroll_end_timer = QtCore.QTimer(self)
        self.scroll_end_timer.setSingleShot(True)
        self.scroll_end_timer.timeout.connect(self.finalize_scroll_session)

        self.auto_timer = QtCore.QTimer(self)
        self.auto_timer.setInterval(3000)  # every 3 seconds
        self.auto_timer.timeout.connect(self.auto_update_speed)
        self.auto_timer.start()

        # For gradual updates to avoid pops:
        self.gradual_timer = QtCore.QTimer(self)
        self.gradual_timer.setInterval(20)  # 20 ms per step
        self.gradual_timer.timeout.connect(self.update_speed_step)
        self.gradual_steps = 0
        self.gradual_current_step = 0
        self.gradual_increment = 0

    def start_gradual_update(self, new_speed, steps=10):
        """Start a gradual update from current smoothed_speed to new_speed in a given number of steps."""
        self.gradual_steps = steps
        self.gradual_current_step = 0
        diff = new_speed - self.smoothed_speed
        self.gradual_increment = diff / steps
        self.gradual_timer.start()
        print(f"Starting gradual update from {self.smoothed_speed:.2f} to {new_speed:.2f} over {steps} steps.")

    def update_speed_step(self):
        """Update one step in the gradual speed change."""
        if self.gradual_current_step >= self.gradual_steps:
            self.gradual_timer.stop()
            return
        self.smoothed_speed += self.gradual_increment
        self.target_speed = self.smoothed_speed
        speed_queue.put(self.smoothed_speed)
        self.label.setText(f"Playback Speed: {self.smoothed_speed:.2f}x")
        self.gradual_current_step += 1

    def toggle_pause(self):
        """Manually toggle the paused state."""
        global paused
        paused = not paused
        if paused:
            self.pause_button.setText("Resume")
            self.label.setText("Playback Paused")
            while True:
                try:
                    audio_queue.get_nowait()
                except Empty:
                    break
            print("Manually paused playback.")
        else:
            self.pause_button.setText("Pause")
            print("Resumed playback.")

    def wheelEvent(self, event):
        """On wheel event: update speed and record session events."""
        global current_speed, paused
        delta = event.angleDelta().y() / 120.0
        now = time.time()

        if not self.scroll_session_active:
            self.scroll_session_active = True
            self.scroll_events.append((now, self.smoothed_speed))
            print(f"Scroll session started, start speed: {self.smoothed_speed:.2f}")

        self.target_speed += delta * 0.1
        self.target_speed = max(MIN_SPEED, min(self.target_speed, MAX_SPEED))
        # Immediate update with exponential smoothing:
        self.smoothed_speed = (SMOOTHING_ALPHA * self.target_speed +
                               (1 - SMOOTHING_ALPHA) * self.smoothed_speed)
        current_speed = self.smoothed_speed

        # If new input brings speed above zero, resume if paused.
        if self.smoothed_speed > 0:
            paused = False
            self.pause_button.setText("Pause")
        speed_queue.put(self.smoothed_speed)
        self.label.setText(f"Playback Speed: {self.smoothed_speed:.2f}x")
        print(f"Immediate speed changed to {self.smoothed_speed:.2f}x via scroll event")
        self.scroll_end_timer.start(300)

    def finalize_scroll_session(self):
        """Record the end of a scroll session."""
        now = time.time()
        self.scroll_events.append((now, self.smoothed_speed))
        print(f"Scroll session ended, end speed: {self.smoothed_speed:.2f}")
        self.scroll_session_active = False

    def auto_update_speed(self):
        """
        Every 3 seconds, if no scroll event occurred, update speed automatically.
        If manually paused, auto-update is skipped.
        Uses gradual update to avoid abrupt changes (and popping).
        """
        global current_speed, paused
        now = time.time()
        epsilon = 1e-6

        if paused:
            print("Auto-update skipped because playback is manually paused.")
            return

        if self.scroll_events:
            most_recent = max(t for t, s in self.scroll_events)
        else:
            most_recent = 0

        if now - most_recent < 3:
            return

        if self.scroll_events:
            recent_events = [(t, s) for (t, s) in self.scroll_events if now - t <= TIME]
            if recent_events:
                if use_gaussian:
                    avg_speed = gaussian_weighted_average(recent_events, now, mu=0, sigma=TIME/2.0, epsilon=epsilon)
                else:
                    weighted_sum = sum(s * (1 / (now - t + epsilon)) for (t, s) in recent_events)
                    total_weight = sum(1 / (now - t + epsilon) for (t, s) in recent_events)
                    avg_speed = weighted_sum / (total_weight + epsilon)
            else:
                avg_speed = self.smoothed_speed
            self.scroll_events.clear()
            new_speed = avg_speed
        else:
            recovery_rate = 0.3
            new_speed = self.smoothed_speed * (1 - recovery_rate)

        if new_speed < 0.07:
            new_speed = 0
            self.label.setText("Playback Paused")
            print("Auto-updated speed below threshold; pausing playback.")
            while True:
                try:
                    audio_queue.get_nowait()
                except Empty:
                    break
            paused = True
        else:
            self.label.setText(f"Playback Speed: {new_speed:.2f}x (Auto-updated)")
            print(f"Auto-updated speed to {new_speed:.2f}x")
            paused = False

        # Instead of updating abruptly, start a gradual update if there is a significant change.
        if abs(new_speed - self.smoothed_speed) > 0.05:
            self.start_gradual_update(new_speed, steps=10)
        else:
            # If change is minimal, update directly.
            self.target_speed = new_speed
            self.smoothed_speed = new_speed
            current_speed = new_speed
            speed_queue.put(new_speed)

    def close_app(self):
        """Stop all processing and quit the application."""
        print("Quit button pressed. Shutting down...")
        stop_event.set()
        while True:
            try:
                audio_queue.get_nowait()
            except Empty:
                break
        audio_queue.put(None)
        self.close()
        QtWidgets.QApplication.quit()

# ------------------------------
# Main Function
# ------------------------------
def main():
    processing_thread = Thread(target=load_and_process_audio, daemon=True)
    playback_thread = Thread(target=audio_playback_worker, daemon=True)
    processing_thread.start()
    playback_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    window = SpeedControlWindow()
    window.show()
    app.exec_()

    processing_thread.join()
    playback_thread.join()
    print("Playback stopped.")

if __name__ == "__main__":
    main()
