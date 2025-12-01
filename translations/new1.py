import torch
import numpy as np
import sounddevice as sd
import queue
import threading
from faster_whisper import WhisperModel
from Levenshtein import ratio  # More effective similarity check

class RealTimeSpeechRecognition:
    def __init__(self, model_size='large-v1', device='cuda', sample_rate=16000, block_duration=1):
        """
        Initialize real-time speech recognition system with optimized recursion reduction
        """
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = 'cpu'

        self.audio_queue = queue.Queue()
        self.recording = False

        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.block_size = int(sample_rate * block_duration)

        try:
            print(f"Loading Whisper {model_size} model on {device}...")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type='float16' if device == 'cuda' else 'float32'
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

        self.current_transcription = ""
        self.transcription_lock = threading.Lock()
        self.previous_segments = []
        self.unique_segments = set()

    def audio_callback(self, indata, frames, time, status):
        """
        Callback function to handle incoming audio data
        """
        if status:
            print(status)

        audio_data = indata[:, 0].copy()
        self.audio_queue.put(audio_data)

    def start_recording(self):
        """
        Start microphone recording
        """
        self.recording = True
        self.current_transcription = ""
        self.previous_segments = []
        self.unique_segments = set()

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.block_size
        )
        self.stream.start()

        self.transcription_thread = threading.Thread(target=self.transcribe_audio)
        self.transcription_thread.start()

        print("Recording started. Speak into the microphone.")

    def stop_recording(self):
        """
        Stop microphone recording
        """
        self.recording = False

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()

        print("\nFinal Transcription:")
        print(self.current_transcription)

        with open('transcription.txt', 'w', encoding='utf-8') as f:
            f.write(self.current_transcription)
        print("Transcription saved to transcription.txt")

    def is_similar_segment(self, new_segment, threshold=0.85):
        """
        Check if a new segment is too similar to previous ones (to reduce repetition)
        """
        for existing_segment in self.previous_segments:
            similarity = ratio(new_segment.lower(), existing_segment.lower())
            if similarity > threshold:
                return True
        return False

    def clean_segment(self, segment):
        """
        Clean and normalize the segment to remove common noise
        """
        segment = ' '.join(segment.split())
        filler_words = ['um', 'uh', 'like', 'you know', 'so', 'well']
        for word in filler_words:
            segment = segment.replace(word, '').strip()
        return segment

    def transcribe_audio(self):
        """
        Continuous transcription of audio from queue with enhanced duplicate filtering
        """
        accumulated_audio = []

        while self.recording:
            try:
                while not self.audio_queue.empty():
                    audio_block = self.audio_queue.get_nowait()
                    accumulated_audio.extend(audio_block)

                if len(accumulated_audio) >= self.sample_rate * 3:  # Minimum 3 seconds
                    audio_np = np.array(accumulated_audio, dtype=np.float32)

                    segments, info = self.model.transcribe(
                        audio_np,
                        beam_size=5,
                        condition_on_previous_text=False,  # Avoid past dependency
                        temperature=0.2,  # Reduce randomness
                        best_of=5,
                        patience=1.0
                    )

                    with self.transcription_lock:
                        for segment in segments:
                            cleaned_text = self.clean_segment(segment.text)

                            if not cleaned_text or self.is_similar_segment(cleaned_text):
                                continue

                            self.unique_segments.add(cleaned_text)
                            self.previous_segments.append(cleaned_text)

                            if len(self.previous_segments) > 10:
                                self.previous_segments.pop(0)

                            self.current_transcription += " " + cleaned_text
                            print(f"Partial Transcription: {cleaned_text}")

                    accumulated_audio.clear()  # Properly reset audio buffer

            except Exception as e:
                print(f"Transcription error: {e}")

            sd.sleep(100)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    try:
        recognizer = RealTimeSpeechRecognition(
            model_size='large-v1',
            device=device
        )

        recognizer.start_recording()

        input("Press Enter to stop recording...\n")

        recognizer.stop_recording()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()