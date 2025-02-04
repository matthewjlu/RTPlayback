import sys
import os
import wave
import tempfile
import threading
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

######################################
# Global Variables / Shared State
######################################
CURRENT_SPEED = 1.0
STOP_PLAYBACK = False

######################################
# Helper Functions
######################################
def text_to_mp3(text, mp3_path):
    """Convert the given text to an MP3 file using gTTS."""
    tts = gTTS(text=text)
    tts.save(mp3_path)

def mp3_to_wav(mp3_path, wav_path, channels=1, sample_rate=22050):
    """
    Convert an MP3 file to a WAV file using pydub.
    Force number of channels and sample rate if specified.
    """
    audio = AudioSegment.from_mp3(mp3_path)
    if audio.channels != channels:
        audio = audio.set_channels(channels)
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)
    audio.export(wav_path, format="wav")


######################################
# Audio Playback Thread
######################################
class AudioPlaybackThread(QThread):
    """
    A QThread that streams audio, applying real-time speed changes
    via pyrubberband, based on the CURRENT_SPEED global variable.
    """
    # Signal to notify MainWindow that playback truly finished
    playback_finished = pyqtSignal()

    def __init__(self, audio_data, rate, channels, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.rate = rate
        self.channels = channels

    def run(self):
        global CURRENT_SPEED, STOP_PLAYBACK
        STOP_PLAYBACK = False  # Reset stop flag each run

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.rate,
                            output=True)
        except Exception as e:
            print(f"Error opening PyAudio stream: {e}")
            self.playback_finished.emit()
            return

        CHUNK_SIZE = 16384 * 2
        index = 0
        total_length = len(self.audio_data)

        while not STOP_PLAYBACK and index < total_length:
            start_time = time.time()

            # Grab a chunk
            chunk_end = min(index + CHUNK_SIZE, total_length)
            chunk_data = self.audio_data[index:chunk_end]
            index = chunk_end

            # Convert chunk to float in [-1, 1]
            # shape: (samples, channels) or (samples, 1)
            chunk_data_t = chunk_data.T  # shape: (channels, samples)
            float_chunk = chunk_data_t.astype(np.float32) / 32768.0

            # Mono or stereo check
            if self.channels == 1:
                float_chunk = float_chunk[0]  # shape: (samples,)

            # Speed stretch
            speed = CURRENT_SPEED
            if abs(speed - 1.0) > 1e-3:
                try:
                    out_chunk_t = pyrb.time_stretch(float_chunk, self.rate, speed)
                except Exception as e:
                    print(f"Error during time-stretch: {e}")
                    out_chunk_t = float_chunk
            else:
                out_chunk_t = float_chunk

            # Convert back to int16
            out_chunk_t = np.clip(out_chunk_t * 32768.0, -32768, 32767).astype(np.int16)

            # Prepare chunk for playback
            if self.channels == 1:
                out_chunk = out_chunk_t.tobytes()  # shape: (samples,)
            else:
                out_chunk = out_chunk_t.T.tobytes() # shape: (samples, channels)

            # Write chunk to stream
            try:
                stream.write(out_chunk)
            except Exception as e:
                print(f"Error writing to stream: {e}")
                break

            # Optional: measure processing time
            end_time = time.time()
            processing_time = end_time - start_time
            chunk_duration = CHUNK_SIZE / self.rate

            # --- Remove or comment out these if they're too noisy ---
            if processing_time > chunk_duration:
                print(f"Warning: processing time {processing_time:.3f}s "
                      f"exceeds chunk duration {chunk_duration:.3f}s")

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Playback thread reached the end (or STOP_PLAYBACK).")
        self.playback_finished.emit()


######################################
# Main Window (PyQt5)
######################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Speed Control (PyQt5)")

        # Layout widgets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.slider_label = QLabel("Playback Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(25)   # 0.25 * 100
        self.speed_slider.setMaximum(300)  # 3.00 * 100
        self.speed_slider.setValue(100)    # 1.00 * 100 by default
        self.speed_slider.setSingleStep(25)  # increments of 0.25
        self.speed_slider.valueChanged.connect(self.on_slider_value_changed)

        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.exit_button = QPushButton("Exit")

        # Connect buttons
        self.play_button.clicked.connect(self.on_play)
        self.stop_button.clicked.connect(self.on_stop)
        self.exit_button.clicked.connect(self.close)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.slider_label)
        main_layout.addWidget(self.speed_slider)
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)

        # Audio data placeholders
        self.audio_data = None
        self.rate = None
        self.channels = None

        # Thread
        self.audio_thread = None

        # Initialize audio in background
        self.init_audio()

    def init_audio(self):
        """
        Create a temporary MP3/WAV file from sample text, 
        load into NumPy array, store in self.audio_data, etc.
        """
        text = (
            "Los Angeles Fire Department Chief Kristin Crowley said a large amount of unburned, dry fuel combined with"
            "low humidity and the expected return of the harsh Santa Ana winds next week could bring more devastation. "
            "She urged residents to clear all brush within 200 feet of their homes." 
            "Flying embers from a wildfire can destroy homes over a mile away," 
            "Crowley said at a briefing Thursday. She asked residents to provide first responders with a "
            "fighting chance to save homes if the fires spread."
        )

        # Use a temporary directory
        self._tmp_dir = tempfile.TemporaryDirectory()
        mp3_path = os.path.join(self._tmp_dir.name, "output.mp3")
        wav_path = os.path.join(self._tmp_dir.name, "output.wav")

        try:
            # Convert text to MP3
            text_to_mp3(text, mp3_path)

            # Convert MP3 to WAV
            mp3_to_wav(mp3_path, wav_path, channels=1, sample_rate=22050)

            # Read WAV
            with wave.open(wav_path, "rb") as wf:
                self.channels = wf.getnchannels()
                self.rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)

            # Reshape audio to (samples, channels)
            if self.channels > 1:
                audio_data = audio_data.reshape(-1, self.channels)
            else:
                audio_data = audio_data.reshape(-1, 1)

            self.audio_data = audio_data

        except Exception as e:
            print(f"Error during audio initialization: {e}")

    def on_slider_value_changed(self, value):
        """
        Slider value changes in increments of 25 = 0.25,
        e.g. value=100 => speed=1.0, value=125 => speed=1.25
        """
        global CURRENT_SPEED
        CURRENT_SPEED = value / 100.0
        self.slider_label.setText(f"Playback Speed: {CURRENT_SPEED:.2f}x")

    def on_play(self):
        global STOP_PLAYBACK

        if self.audio_data is None:
            print("No audio data loaded.")
            return

        # If a thread is already running, don't start again
        if self.audio_thread is not None and self.audio_thread.isRunning():
            print("Playback is already running.")
            return

        # Create a new thread for each playback session
        self.audio_thread = AudioPlaybackThread(
            audio_data=self.audio_data,
            rate=self.rate,
            channels=self.channels
        )
        self.audio_thread.playback_finished.connect(self.on_playback_finished)
        self.audio_thread.start()
        print("Playback started...")

    def on_stop(self):
        global STOP_PLAYBACK
        STOP_PLAYBACK = True  # The thread will end on its own
        print("Stop signal sent.")

    def on_playback_finished(self):
        """
        Called when the QThread signals it has stopped 
        (either because audio ended or STOP_PLAYBACK was set).
        """
        print("Playback finished in thread.")
        # Do NOT set the thread to None if you want 
        # to read debug info after finishing. If you do want
        # to allow re-Play, you can safely set it to None here:
        self.audio_thread = None
        # Now the user can press "Play" again and 
        # a new thread will be created.

    def closeEvent(self, event):
        """
        Overriding closeEvent to stop playback gracefully when window closes.
        """
        global STOP_PLAYBACK
        STOP_PLAYBACK = True

        # If a thread is running, wait for it to finish
        if self.audio_thread is not None and self.audio_thread.isRunning():
            print("Waiting for playback thread to finish...")
            self.audio_thread.wait()  # wait indefinitely until finished

        event.accept()


######################################
# Application Entry Point
######################################
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(400, 200)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
