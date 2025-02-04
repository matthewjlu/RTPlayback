import sys
import os
import wave
import tempfile
import queue
import time

import numpy as np
import pyaudio
import pyrubberband as pyrb

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QSlider, QPushButton, QLabel
)

from gtts import gTTS
from pydub import AudioSegment

############################
# Global / Shared Variables
############################
CURRENT_SPEED = 1.0
STOP_FLAG = False  # We'll use a shared flag to stop both threads.


############################
# Helper Functions
############################
def text_to_mp3(text, mp3_path):
    tts = gTTS(text=text)
    tts.save(mp3_path)

def mp3_to_wav(mp3_path, wav_path, channels=1, sample_rate=22050):
    audio = AudioSegment.from_mp3(mp3_path)
    if audio.channels != channels:
        audio = audio.set_channels(channels)
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    audio.export(wav_path, format="wav")


############################
# Producer Thread
############################
class ProducerThread(QThread):
    """
    Continuously reads small chunks from the original audio_data,
    applies time-stretch at the CURRENT_SPEED, and places them
    into a queue for the consumer to play.
    """

    def __init__(self, audio_data, rate, channels, q, parent=None):
        """
        audio_data: original int16 array, shape (samples, channels)
        rate, channels: int
        q: queue.Queue for PCM data
        """
        super().__init__(parent)
        self.audio_data = audio_data
        self.rate = rate
        self.channels = channels
        self.q = q

    def run(self):
        global CURRENT_SPEED, STOP_FLAG

        CHUNK_SIZE = 4096  # or 8192, experiment with sizes
        idx = 0
        total_len = len(self.audio_data)

        while not STOP_FLAG and idx < total_len:
            # Grab a chunk of raw samples (int16)
            end = min(idx + CHUNK_SIZE, total_len)
            raw_chunk = self.audio_data[idx:end]
            idx = end

            # Convert to float, shape: (samples, channels) => transpose => (channels, samples)
            chunk_t = raw_chunk.T.astype(np.float32) / 32768.0

            # For mono, shape: (1, samples), so let's flatten to (samples,)
            if self.channels == 1:
                chunk_t = chunk_t[0]  # shape: (samples,)

            # Apply time stretch using the *current* slider speed
            speed = CURRENT_SPEED
            if abs(speed - 1.0) > 1e-3:
                try:
                    stretched = pyrb.time_stretch(chunk_t, self.rate, speed)
                except Exception as e:
                    print(f"Time-stretch error: {e}")
                    stretched = chunk_t
            else:
                stretched = chunk_t

            # Convert back to int16
            stretched_int16 = np.clip(stretched * 32768.0, -32768, 32767).astype(np.int16)

            # If mono, shape: (samples,) => reshape to (samples,1)
            # If stereo, shape: (2, samples) => we might need to transpose back
            if self.channels == 1:
                stretched_int16 = stretched_int16.reshape(-1, 1)
            else:
                # If stereo, shape: (samples,2) is typical, but rubberband might return (2, samples)
                # So let's check and transpose if needed:
                if stretched_int16.ndim == 2 and stretched_int16.shape[0] == self.channels:
                    stretched_int16 = stretched_int16.T

            # Put the stretched chunk into the queue
            self.q.put(stretched_int16)

            # If the queue gets too big, you can optionally do a .put call with block=False
            # or limit the size with qsize checks

        # Signal we've reached the end of the file
        print("ProducerThread finished or STOP_FLAG triggered.")


############################
# Consumer Thread
############################
class ConsumerThread(QThread):
    """
    Reads int16 PCM chunks from the queue and writes them
    to a PyAudio stream for playback.
    """
    playback_finished = pyqtSignal()  # Emitted when we exhaust the queue or STOP_FLAG is set

    def __init__(self, rate, channels, q, parent=None):
        super().__init__(parent)
        self.rate = rate
        self.channels = channels
        self.q = q

    def run(self):
        global STOP_FLAG

        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.rate,
                            output=True)
        except Exception as e:
            print(f"Error opening stream: {e}")
            self.playback_finished.emit()
            return

        # We'll read from the queue until the queue is empty AND the producer is done
        # or we see STOP_FLAG is True. A simple approach is to keep reading
        # until STOP_FLAG or we can't get data from the queue for a while.
        while not STOP_FLAG or not self.q.empty():
            try:
                # If queue is empty, this will block a bit
                chunk_int16 = self.q.get(timeout=0.1)
            except queue.Empty:
                # If empty, check again if we should continue
                if STOP_FLAG:
                    break
                continue

            # Convert the chunk to bytes
            out_bytes = chunk_int16.tobytes()

            try:
                stream.write(out_bytes)
            except Exception as e:
                print(f"Error writing to PyAudio: {e}")
                break

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("ConsumerThread finished.")
        self.playback_finished.emit()


############################
# MainWindow (PyQt5)
############################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Audio Speed Control (Producer/Consumer)")

        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.slider_label = QLabel("Playback Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(25)   # 0.25 * 100
        self.speed_slider.setMaximum(300)  # 3.00 * 100
        self.speed_slider.setValue(100)    # 1.00 * 100
        self.speed_slider.setSingleStep(25)
        self.speed_slider.valueChanged.connect(self.on_slider_value_changed)

        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.exit_button = QPushButton("Exit")

        self.play_button.clicked.connect(self.on_play)
        self.stop_button.clicked.connect(self.on_stop)
        self.exit_button.clicked.connect(self.close)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.slider_label)
        main_layout.addWidget(self.speed_slider)
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)

        # Audio data
        self.audio_data = None
        self.rate = None
        self.channels = None

        # Threads
        self.producer_thread = None
        self.consumer_thread = None

        # The queue for PCM chunks
        self.q = queue.Queue(maxsize=10)  # or None for unlimited

        # Initialize the audio file
        self.init_audio()

    def init_audio(self):
        """
        Convert sample text to MP3->WAV->NumPy and store in self.audio_data.
        """
        text = (
            "Los Angeles Fire Department Chief Kristin Crowley said a large amount of unburned, dry fuel combined with"
            "low humidity and the expected return of the harsh Santa Ana winds next week could bring more devastation. "
            "She urged residents to clear all brush within 200 feet of their homes." 
            "Flying embers from a wildfire can destroy homes over a mile away," 
            "Crowley said at a briefing Thursday. She asked residents to provide first responders with a "
            "fighting chance to save homes if the fires spread."
        )

        tmp_dir = tempfile.TemporaryDirectory()
        mp3_path = os.path.join(tmp_dir.name, "output.mp3")
        wav_path = os.path.join(tmp_dir.name, "output.wav")

        try:
            text_to_mp3(text, mp3_path)
            mp3_to_wav(mp3_path, wav_path, channels=1, sample_rate=22050)

            with wave.open(wav_path, "rb") as wf:
                self.channels = wf.getnchannels()
                self.rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

            if self.channels > 1:
                audio_data = audio_data.reshape(-1, self.channels)
            else:
                audio_data = audio_data.reshape(-1, 1)

            self.audio_data = audio_data

        except Exception as e:
            print(f"Audio init error: {e}")

        # Keep ref so temp files don't vanish:
        self._tmp_dir = tmp_dir

    def on_slider_value_changed(self, value):
        global CURRENT_SPEED
        CURRENT_SPEED = value / 100.0
        self.slider_label.setText(f"Playback Speed: {CURRENT_SPEED:.2f}x")

    def on_play(self):
        global STOP_FLAG
        if self.audio_data is None:
            print("No audio loaded.")
            return

        # If already playing, ignore
        if (self.producer_thread and self.producer_thread.isRunning()) or \
           (self.consumer_thread and self.consumer_thread.isRunning()):
            print("Playback is already running.")
            return

        # Reset the stop flag and queue
        STOP_FLAG = False
        # Clear any old data in the queue
        with self.q.mutex:
            self.q.queue.clear()

        # Start producer
        self.producer_thread = ProducerThread(
            audio_data=self.audio_data,
            rate=self.rate,
            channels=self.channels,
            q=self.q
        )
        self.producer_thread.start()

        # Start consumer
        self.consumer_thread = ConsumerThread(
            rate=self.rate,
            channels=self.channels,
            q=self.q
        )
        self.consumer_thread.playback_finished.connect(self.on_playback_finished)
        self.consumer_thread.start()

        print("Playback started with real-time speed control.")

    def on_stop(self):
        global STOP_FLAG
        STOP_FLAG = True
        print("Stop signal sent.")

    def on_playback_finished(self):
        print("Playback finished (Consumer thread).")
        # Optionally set threads to None, or do more cleanup if needed
        self.producer_thread = None
        self.consumer_thread = None

    def closeEvent(self, event):
        """
        Called when the window closes. Stop threads gracefully.
        """
        global STOP_FLAG
        STOP_FLAG = True

        # Wait for threads if they're running
        if self.producer_thread and self.producer_thread.isRunning():
            self.producer_thread.wait()
        if self.consumer_thread and self.consumer_thread.isRunning():
            self.consumer_thread.wait()

        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(400, 200)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
