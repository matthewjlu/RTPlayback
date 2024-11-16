import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
from gtts import gTTS
from tempfile import NamedTemporaryFile
import vlc

class TTSApp(QWidget):
    speech_ready_signal = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()
        self.initUI()

        self.temp_file = None

        # VLC player
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()

        # Speed control parameters
        self.current_speed = 1.0
        self.target_speed = 1.0
        self.min_speed = 0.25   # Minimum speed VLC can handle
        self.max_speed = 2.0

        # Volume control parameters
        self.current_volume = 100  # VLC volume ranges from 0 to 100
        self.target_volume = 100

        # Smoothing parameters
        self.speed_smoothing_factor = 0.05  # Adjust for smoother speed transitions
        self.volume_smoothing_factor = 0.1  # Adjust for smoother volume transitions
        self.decay_rate = 0.002             # Adjust if needed

        # Timers and flags
        self.speed_update_timer = QTimer()
        self.speed_update_timer.setInterval(50)  # Increased interval for smoother updates
        self.speed_update_timer.timeout.connect(self.update_playback_speed)

        self.inactivity_timer = QTimer()
        self.inactivity_timer.setSingleShot(True)
        self.inactivity_timer.setInterval(10000)
        self.inactivity_timer.timeout.connect(self.start_speed_decay)

        self.decaying = False

        # Text input handling
        self.text_change_timer = QTimer()
        self.text_change_timer.setSingleShot(True)
        self.text_change_timer.setInterval(1000)
        self.text_change_timer.timeout.connect(self.on_text_input_finished)

        self.text_edit.textChanged.connect(self.on_text_changed)
        self.prepared_text = ''
        self.speech_ready = False
        self.speech_task_id = 0

        self.speech_ready_signal.connect(self.on_speech_ready)

        # Loading animation
        self.loading_animation_timer = QTimer()
        self.loading_animation_timer.setInterval(300)
        self.loading_animation_timer.timeout.connect(self.update_loading_animation)
        self.loading_dots = 0

        # Connect VLC media end event
        self.vlc_player.event_manager().event_attach(vlc.EventType.MediaPlayerEndReached, self.on_media_end)

    def initUI(self):
        self.text_edit = QTextEdit(self)
        self.speak_button = QPushButton('Speak', self)
        self.speak_button.setEnabled(False)
        self.pause_button = QPushButton('Pause', self)
        self.speed_label = QLabel('Playback Speed: 1.00x', self)

        self.speed_control_widget = QLabel('Scroll here to adjust playback speed', self)
        self.speed_control_widget.setAlignment(Qt.AlignCenter)
        self.speed_control_widget.setStyleSheet('background-color: lightgray;')
        self.speed_control_widget.setFixedHeight(50)

        self.speed_control_widget.installEventFilter(self)

        hbox = QHBoxLayout()
        hbox.addWidget(self.speak_button)
        hbox.addWidget(self.pause_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.text_edit)
        vbox.addLayout(hbox)
        vbox.addWidget(self.speed_label)
        vbox.addWidget(self.speed_control_widget)
        self.setLayout(vbox)

        self.speak_button.clicked.connect(self.on_speak)
        self.pause_button.clicked.connect(self.on_pause)

        self.setWindowTitle('TTS Scroll Speed Control')
        self.setGeometry(100, 100, 400, 400)
        self.show()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel and source == self.speed_control_widget:
            delta = event.angleDelta().y()
            self.adjust_target_speed(delta)

            if not self.speed_update_timer.isActive():
                self.speed_update_timer.start()

            self.inactivity_timer.start()
            self.decaying = False

            return True
        return super().eventFilter(source, event)

    def adjust_target_speed(self, delta):
        scaling_factor = 0.002
        self.target_speed += delta * scaling_factor
        self.target_speed = max(0.01, min(self.target_speed, self.max_speed))

        # Adjust target volume based on speed
        if self.target_speed <= self.min_speed:
            # Calculate volume proportionally to speed
            self.target_volume = int((self.target_speed / self.min_speed) * 100)
        else:
            self.target_volume = 100

    def update_playback_speed(self):
        if self.decaying:
            self.target_speed = max(0.01, self.target_speed - self.decay_rate)

            # Adjust target volume during decay
            if self.target_speed <= self.min_speed:
                self.target_volume = int((self.target_speed / self.min_speed) * 100)
            else:
                self.target_volume = 100

        # Exponential smoothing for speed
        self.current_speed = (self.speed_smoothing_factor * self.target_speed) + \
                             ((1 - self.speed_smoothing_factor) * self.current_speed)
        self.current_speed = max(self.min_speed, min(self.current_speed, self.max_speed))

        # Exponential smoothing for volume
        self.current_volume = (self.volume_smoothing_factor * self.target_volume) + \
                              ((1 - self.volume_smoothing_factor) * self.current_volume)
        self.current_volume = max(0, min(int(self.current_volume), 100))

        # Update VLC playback rate and volume
        if self.vlc_player.is_playing():
            if abs(self.vlc_player.get_rate() - self.current_speed) > 0.01:
                self.vlc_player.set_rate(self.current_speed)
            if abs(self.vlc_player.audio_get_volume() - self.current_volume) > 1:
                self.vlc_player.audio_set_volume(self.current_volume)

        self.speed_label.setText(f'Playback Speed: {self.current_speed:.2f}x')

        # Stop updates when target is reached
        if abs(self.current_speed - self.target_speed) < 0.001 and \
           abs(self.current_volume - self.target_volume) < 1:
            self.current_speed = self.target_speed
            self.current_volume = self.target_volume
            if self.decaying and self.current_speed <= 0.01:
                self.decaying = False
                self.speed_update_timer.stop()
                # Optionally stop playback when volume reaches zero
                self.vlc_player.pause()
            elif not self.decaying:
                self.speed_update_timer.stop()

    def start_speed_decay(self):
        if self.vlc_player.is_playing():
            self.decaying = True
            if not self.speed_update_timer.isActive():
                self.speed_update_timer.start()

    def on_text_changed(self):
        self.text_change_timer.start()
        self.speech_ready = False
        self.speak_button.setEnabled(False)
        self.speak_button.setText('Speak')

    def on_text_input_finished(self):
        text = self.text_edit.toPlainText()
        if text.strip() == '':
            return
        self.prepared_text = text
        self.speech_task_id += 1
        current_task_id = self.speech_task_id
        self.generate_speech(current_task_id)

    def generate_speech(self, task_id):
        self.loading_dots = 0
        self.loading_animation_timer.start()
        threading.Thread(target=self.synthesize_speech, args=(self.prepared_text, task_id)).start()

    def update_loading_animation(self):
        dots = '.' * (self.loading_dots % 4)
        self.speak_button.setText(f'Loading{dots}')
        self.loading_dots += 1

    def synthesize_speech(self, text, task_id):
        try:
            tts = gTTS(text=text, lang='en')
            temp_file = NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            temp_file.close()
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            temp_file = None
        self.speech_ready_signal.emit(task_id, temp_file)

    def on_speech_ready(self, task_id, temp_file):
        self.loading_animation_timer.stop()
        if task_id != self.speech_task_id:
            if temp_file:
                os.unlink(temp_file.name)
            return

        if temp_file is None:
            self.speak_button.setText('Error')
            return

        self.cleanup_temp_file()
        self.temp_file = temp_file
        self.speech_ready = True
        self.speak_button.setEnabled(True)
        self.speak_button.setText('Speak')

        # Set up VLC media player
        media = self.vlc_instance.media_new(self.temp_file.name)
        self.vlc_player.set_media(media)

    def on_speak(self):
        if not self.speech_ready:
            print("Speech not ready yet. Please wait.")
            return

        if self.vlc_player.is_playing():
            # Already playing
            return

        self.vlc_player.play()
        self.vlc_player.audio_set_volume(self.current_volume)
        self.inactivity_timer.start()
        if not self.speed_update_timer.isActive():
            self.speed_update_timer.start()

    def on_pause(self):
        if self.vlc_player.is_playing():
            self.vlc_player.pause()
            self.inactivity_timer.stop()
            self.decaying = False
        else:
            self.vlc_player.play()
            self.inactivity_timer.start()
            if not self.speed_update_timer.isActive():
                self.speed_update_timer.start()

    def on_media_end(self, event):
        self.cleanup_temp_file()
        self.speech_ready = False
        self.vlc_player.stop()
        self.speed_update_timer.stop()
        self.inactivity_timer.stop()
        self.decaying = False

    def cleanup_temp_file(self):
        if self.temp_file:
            try:
                os.unlink(self.temp_file.name)
            except Exception as e:
                print(f"Error deleting temp file: {e}")
            self.temp_file = None

    def closeEvent(self, event):
        self.cleanup_temp_file()
        self.vlc_player.stop()
        self.vlc_player.release()
        self.vlc_instance.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TTSApp()
    sys.exit(app.exec_())
