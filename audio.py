import pyaudio
import pyrubberband as pyrb
import numpy as np
import librosa
from queue import Queue
from threading import Event, Thread

# ------------------------------
# Configuration
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

def load_and_process_audio():
    """
    1) Loads entire audio.
    2) In a loop, reads BATCH_SIZE samples from 'y', time-stretches them using the
       current speed, then splits into smaller sub-chunks for the audio queue.
    3) Checks for speed changes after each batch.
    """
    # Load/normalize
    y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak  # One-time normalization

    total_samples = len(y)
    position = 0

    # Convert desired batch duration to samples
    BATCH_SIZE = int(BATCH_DURATION * SAMPLE_RATE)
    current_speed = 1.0

    while position < total_samples and not stop_event.is_set():
        # Check for any updated speed:
        if not speed_queue.empty():
            current_speed = speed_queue.get()
            # Clamp to [MIN_SPEED, MAX_SPEED]
            current_speed = max(MIN_SPEED, min(current_speed, MAX_SPEED))

        # Read one "batch" from the original
        end = min(position + BATCH_SIZE, total_samples)
        raw_chunk = y[position:end]
        position = end

        # Time-stretch if needed
        if current_speed != 1.0:
            stretched = pyrb.time_stretch(raw_chunk, sr, current_speed)
        else:
            stretched = raw_chunk

        # Now 'stretched' might be bigger/smaller than BATCH_SIZE
        # Break it into smaller chunks for playback
        idx = 0
        while idx < len(stretched) and not stop_event.is_set():
            sub_end = min(idx + PLAYBACK_CHUNK, len(stretched))
            sub_chunk = stretched[idx:sub_end]
            idx = sub_end

            # Put this small sub-chunk in the queue
            audio_queue.put(sub_chunk)

    # Signal end of audio
    audio_queue.put(None)

def audio_playback_worker():
    """ Continuously fetch frames from the queue and write to PyAudio. """
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

        # Convert chunk to float32 and write
        data = chunk.astype(np.float32).tobytes()
        stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    processing_thread = Thread(target=load_and_process_audio, daemon=True)
    playback_thread = Thread(target=audio_playback_worker, daemon=True)

    processing_thread.start()
    playback_thread.start()

    print("Real-time Speed Control")
    print(f"Enter speed multipliers ({MIN_SPEED}-{MAX_SPEED}), or 'exit':")

    try:
        while processing_thread.is_alive() and playback_thread.is_alive():
            user_input = input("Speed: ").strip().lower()
            if user_input == 'exit':
                stop_event.set()
                break
            try:
                new_speed = float(user_input)
                if MIN_SPEED <= new_speed <= MAX_SPEED:
                    speed_queue.put(new_speed)
                    print(f"Speed changed to {new_speed}x")
                else:
                    print(f"Speed must be between {MIN_SPEED} and {MAX_SPEED}")
            except ValueError:
                print("Invalid input.")
    except KeyboardInterrupt:
        stop_event.set()

    # Wait for threads
    processing_thread.join()
    playback_thread.join()
    print("Playback stopped")

if __name__ == "__main__":
    main()
