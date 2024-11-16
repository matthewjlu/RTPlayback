import sys
import os
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QLabel, QVBoxLayout, QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from gtts import gTTS
from tempfile import NamedTemporaryFile
import vlc

class TTSApp(QWidget):
    speech_ready_signal = pyqtSignal(int, object)

    def __init__(self):
        super().__init__()

        self.temp_file = None

        # VLC player
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()

        # Playback speed parameters
        self.current_speed = 1.0
        self.target_speed = 1.0
        self.min_speed = 0.25   # Minimum speed VLC can handle
        self.max_speed = 2.0

        # Timer for gradual speed adjustment
        self.speed_update_timer = QTimer()
        self.speed_update_timer.setInterval(400)  # Adjust interval for smoothness
        self.speed_update_timer.timeout.connect(self.update_playback_speed)

        self.initUI()

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
        self.speed_label = QLabel(f'Playback Speed: {self.current_speed:.2f}x', self)

        # Slider for playback speed control
        self.speed_slider = QSlider(Qt.Horizontal, self)
        self.speed_slider.setMinimum(int(self.min_speed * 100))
        self.speed_slider.setMaximum(int(self.max_speed * 100))
        self.speed_slider.setValue(int(self.current_speed * 100))
        self.speed_slider.setTickInterval(25)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.valueChanged.connect(self.on_speed_slider_changed)

        hbox = QHBoxLayout()
        hbox.addWidget(self.speak_button)
        hbox.addWidget(self.pause_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.text_edit)
        vbox.addLayout(hbox)
        vbox.addWidget(self.speed_label)
        vbox.addWidget(self.speed_slider)
        self.setLayout(vbox)

        self.speak_button.clicked.connect(self.on_speak)
        self.pause_button.clicked.connect(self.on_pause)

        self.setWindowTitle('TTS Playback Speed Control')
        self.setGeometry(100, 100, 400, 400)
        self.show()

    def on_speed_slider_changed(self, value):
        self.target_speed = value / 100.0
        self.target_speed = max(self.min_speed, min(self.target_speed, self.max_speed))
        self.speed_label.setText(f'Playback Speed: {self.target_speed:.2f}x')
        if self.vlc_player.is_playing():
            # Start or restart the speed update timer
            if not self.speed_update_timer.isActive():
                self.speed_update_timer.start()
        else:
            # If not playing, set current speed directly
            self.current_speed = self.target_speed
            self.vlc_player.set_rate(self.current_speed)

    def update_playback_speed(self):
        # Gradually adjust the current speed towards the target speed
        speed_difference = self.target_speed - self.current_speed
        speed_step = 0.01  # Adjust this value for smoothness and speed of transition

        if abs(speed_difference) < 0.01:
            # Close enough to target speed
            self.current_speed = self.target_speed
            self.speed_update_timer.stop()
        else:
            # Increase or decrease current speed towards target speed
            self.current_speed += speed_step if speed_difference > 0 else -speed_step
            # Ensure current_speed stays within min and max limits
            self.current_speed = max(self.min_speed, min(self.current_speed, self.max_speed))

        # Update VLC playback rate
        self.vlc_player.set_rate(self.current_speed)
        self.speed_label.setText(f'Playback Speed: {self.current_speed:.2f}x')

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
        self.vlc_player.set_rate(self.current_speed)

    def on_speak(self):
        if not self.speech_ready:
            print("Speech not ready yet. Please wait.")
            return

        if self.vlc_player.is_playing():
            # Already playing
            return

        self.vlc_player.play()
        self.vlc_player.audio_set_volume(100)  # Set volume to maximum
        self.vlc_player.set_rate(self.current_speed)

    def on_pause(self):
        if self.vlc_player.is_playing():
            self.vlc_player.pause()
        else:
            self.vlc_player.play()
            self.vlc_player.set_rate(self.current_speed)

    def on_media_end(self, event):
        self.cleanup_temp_file()
        self.speech_ready = False
        self.vlc_player.stop()
        self.speed_update_timer.stop()  # Stop the timer when playback ends

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
