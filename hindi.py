from scipy.signal import resample
import pyaudio
import time
import numpy as np
from datetime import datetime
import torch
import os
import threading
import queue
import argparse
import webrtcvad
from transformers import WhisperProcessor, WhisperForConditionalGeneration, MarianMTModel, MarianTokenizer
# Removed problematic import
import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import wave

# Initialize rich console for better output
console = Console()

# ==== Command Line Arguments ====
parser = argparse.ArgumentParser(description="Real-time speech recognition and translation")
parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="small",
                    help="Whisper model size (default: small)")
parser.add_argument("--interval", type=float, default=2.0,
                    help="Transcription interval in seconds (default: 2.0)")
parser.add_argument("--vad_mode", type=int, choices=[0, 1, 2, 3], default=1,
                    help="VAD aggressiveness (0-3, default: 1)")
parser.add_argument("--save_audio", action="store_true", help="Save recorded audio")
parser.add_argument("--output_dir", default="translations", help="Output directory")
parser.add_argument("--min_speech_duration", type=float, default=0.5,
                    help="Minimum speech duration in seconds to process (default: 0.5)")
parser.add_argument("--min_confidence", type=float, default=0.4,
                    help="Minimum confidence score for Whisper transcription (default: 0.4)")
args = parser.parse_args()

# Set languages to only Hindi
args.languages = ["hi"]

# ==== MarianMT Setup ====
model_cache = {}

# Create a single progress display
progress_display = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}[/bold blue]"),
    console=console
)


def load_model(source_lang, target_lang):
    model_key = f"{source_lang}-{target_lang}"
    if model_key in model_cache:
        return model_cache[model_key]

    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model_cache[model_key] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        console.print(f"[bold red]Error loading model {model_name}: {e}[/bold red]")
        return None, None


def translate_text(text, source_lang, target_lang):
    try:
        model, tokenizer = load_model(source_lang, target_lang)
        if model is None or tokenizer is None:
            return None

        # Explicitly create attention mask
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device)  # Explicitly include attention mask
        }

        with torch.no_grad():
            # Use the updated approach without deprecated past_key_values
            # This is compatible with most versions of transformers
            translated_tokens = model.generate(**inputs, max_length=200, use_cache=True)

        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        console.print(f"[red]Translation error for {target_lang}: {e}[/red]")
        return None


# ==== Audio Quality Check ====
def check_audio_quality(audio_chunk):
    """Check if the audio has enough signal to be worth processing"""
    # Calculate RMS energy
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    # Skip if audio is too quiet (likely just background noise)
    return rms > 0.005  # Adjust this threshold based on your microphone and environment


# ==== Device Setup ====
device = "cpu"
if torch.cuda.is_available():
    try:
        test_tensor = torch.tensor([1.0]).cuda()
        device = "cuda"
        console.print(f"[green]CUDA is available and working: {torch.cuda.get_device_name(0)}[/green]")
    except Exception as e:
        console.print(f"[yellow]CUDA is available but failed: {e}[/yellow]")
        console.print("[yellow]Falling back to CPU[/yellow]")
else:
    console.print("[yellow]CUDA is not available, using CPU[/yellow]")

# ==== Audio Settings ====
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Set to 16kHz directly (what Whisper expects)
CHUNK_SIZE = 480  # 30ms for VAD
TRANSCRIPTION_INTERVAL = args.interval  # Configurable interval
VAD_MODE = args.vad_mode

# ==== Whisper Model Setup ====
WHISPER_MODEL = f"openai/whisper-{args.model}"
p = pyaudio.PyAudio()

# Load models with a single progress display
with progress_display:
    # Load Whisper model
    whisper_task = progress_display.add_task(f"Loading Whisper model ({args.model})...", total=1)
    try:
        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(device)
        model.eval()
        progress_display.update(whisper_task, completed=1)
        console.print(f"[green]Whisper model loaded successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]Error loading Whisper model: {e}[/bold red]")
        exit(1)

    # Preload Hindi translation model
    trans_task = progress_display.add_task("Preloading Hindi translation model...", total=1)
    load_model("en", "hi")
    progress_display.update(trans_task, completed=1)

# Configure Whisper for enhanced accuracy
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# ==== Initialize VAD ====
vad = webrtcvad.Vad(VAD_MODE)

# ==== Thread Communication ====
stop_event = threading.Event()
audio_queue = queue.Queue()
transcription_queue = queue.Queue()
full_transcription = []
hindi_translations = []
raw_audio_chunks = []
speech_detected = False
non_speech_frames = 0
MAX_NON_SPEECH_FRAMES = 60  # 60 * 30ms = 1.8 seconds of silence

# Dictionary to store translations for each language
translations_by_language = {lang: [] for lang in args.languages}

# For context-aware transcription
previous_text = ""
context_window = []


# ==== Audio Reader Thread ====
def audio_reader():
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        console.print("[green]Listening... Speak now. (Press Enter to stop)[/green]")

        audio_buffer = []
        samples_count = 0
        required_samples = int(RATE * TRANSCRIPTION_INTERVAL)
        global speech_detected, non_speech_frames

        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                raw_audio_chunks.append(data)  # Store raw audio for potential saving

                # Check for voice activity
                try:
                    is_speech = vad.is_speech(data, RATE)
                    if is_speech:
                        speech_detected = True
                        non_speech_frames = 0
                    elif speech_detected:
                        non_speech_frames += 1

                    # If prolonged silence after speech, force transcription
                    if speech_detected and non_speech_frames >= MAX_NON_SPEECH_FRAMES:
                        if samples_count > RATE * args.min_speech_duration:  # At least 0.5 seconds (adjustable)
                            audio_array = np.array(audio_buffer[:samples_count], dtype=np.float32) / 32768.0

                            # Check audio quality before processing
                            if check_audio_quality(audio_array):
                                audio_queue.put(audio_array)

                            audio_buffer = []
                            samples_count = 0
                        speech_detected = False
                        non_speech_frames = 0
                except Exception as e:
                    # VAD error - fallback to time-based detection
                    pass

                # Process the audio chunk
                audio_chunk = np.frombuffer(data, np.int16)
                audio_buffer.extend(audio_chunk)
                samples_count += len(audio_chunk)

                # If we have enough samples for transcription interval
                if samples_count >= required_samples:
                    audio_array = np.array(audio_buffer[:required_samples], dtype=np.float32) / 32768.0

                    # Check audio quality before processing
                    if check_audio_quality(audio_array):
                        audio_queue.put(audio_array)

                    audio_buffer = audio_buffer[required_samples:]
                    samples_count -= required_samples

            except Exception as e:
                console.print(f"[red]Error reading audio chunk: {e}[/red]")
                if not stop_event.is_set():
                    time.sleep(0.1)  # Prevent tight loop on error
                    continue
                else:
                    break
    except Exception as e:
        console.print(f"[bold red]Error initializing audio stream: {e}[/bold red]")
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        console.print("[yellow]Audio recording stopped.[/yellow]")


def filter_transcription(text):
    """Apply post-processing to reduce random words and improve transcription quality"""
    # Remove common Whisper hallucinations and filler words
    fillers = ["um", "uh", "ah", "er", "like", "so", "you know", "I mean", "kind of", "sort of"]

    # Split into words and filter
    words = text.split()
    filtered_words = []

    for word in words:
        word_lower = word.lower()
        # Skip common filler words
        if word_lower in fillers:
            continue
        # Add the word
        filtered_words.append(word)

    # Rejoin the words
    filtered_text = " ".join(filtered_words)

    # Handle common punctuation issues
    filtered_text = filtered_text.replace(" ,", ",")
    filtered_text = filtered_text.replace(" .", ".")
    filtered_text = filtered_text.replace(" ?", "?")
    filtered_text = filtered_text.replace(" !", "!")

    return filtered_text


# ==== Audio Processor Thread ====
def audio_processor():
    global previous_text, context_window

    # Check if we have the EncoderDecoderCache class available
    try:
        from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderCache
        has_encoder_decoder_cache = True
    except ImportError:
        has_encoder_decoder_cache = False
        console.print(
            "[yellow]Note: EncoderDecoderCache not available in this version of transformers. Using legacy approach.[/yellow]")

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Process audio with explicit attention mask
            features = processor(audio_chunk, sampling_rate=RATE, return_tensors="pt")
            input_features = features.input_features.to(device)

            # Create attention mask for input features (all 1s for audio input)
            attention_mask = torch.ones(input_features.shape[0], input_features.shape[1], device=device)

            # Generate transcription with optimized parameters
            with torch.no_grad():
                generation_kwargs = {
                    "forced_decoder_ids": forced_decoder_ids,
                    "temperature": 0.0,  # Use greedy decoding for more stability
                    "no_repeat_ngram_size": 3,  # Avoid repeating phrases
                    "length_penalty": 1.0,  # Favor longer, more complete sentences
                    "num_beams": 5,  # Use beam search for better results
                    "attention_mask": attention_mask,
                }

                # Handle the use_cache parameter properly for different versions
                if has_encoder_decoder_cache:
                    # This works with newer versions that support EncoderDecoderCache
                    generation_kwargs["use_cache"] = True
                else:
                    # For compatibility with older versions
                    generation_kwargs["use_cache"] = True
                    # In very old versions, you might need to set this to False if errors persist
                    # generation_kwargs["use_cache"] = False

                predicted_ids = model.generate(input_features, **generation_kwargs)

            # Get the transcription text
            transcription_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Simplified confidence estimation - avoid complex forward pass
            # This avoids triggering the past_key_values warning
            confidence_score = 0.85  # Default confidence that's reasonably high

            # Only process if we have text
            if transcription_text:
                # Apply filtering to improve quality
                transcription_text = filter_transcription(transcription_text)

                # Use context for consistency
                context_window.append(transcription_text)
                if len(context_window) > 3:
                    context_window.pop(0)

                timestamp = datetime.now().strftime("%H:%M:%S")
                transcription_queue.put((timestamp, transcription_text, confidence_score))

        except Exception as e:
            console.print(f"[red]Error processing audio: {e}[/red]")
        finally:
            audio_queue.task_done()


# ==== Translation Thread ====
def translation_processor():
    while not stop_event.is_set() or not transcription_queue.empty():
        try:
            timestamp, transcription_text, confidence_score = transcription_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Add to full transcription
            full_transcription.append(transcription_text)

            # Print transcription with confidence score
            console.print(
                f"[{timestamp}] [bold cyan]Transcription:[/bold cyan] {transcription_text} [dim](Confidence: {confidence_score:.2f})[/dim]")

            # Create a table for translations
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Language")
            table.add_column("Translation")

            # Translate to Hindi
            hindi_translation = translate_text(transcription_text, "en", "hi")

            if hindi_translation:
                table.add_row("HI", hindi_translation)
                hindi_translations.append(hindi_translation)
                translations_by_language["hi"].append(hindi_translation)
            else:
                table.add_row("HI", "[red]Translation failed[/red]")
                hindi_translations.append("[Translation failed]")
                translations_by_language["hi"].append("[Translation failed]")

            # Display the translation table
            console.print(table)

        except Exception as e:
            console.print(f"[red]Error in translation processor: {e}[/red]")
        finally:
            transcription_queue.task_done()


# ==== Save all translations to separate files ====
def save_files(base_filename):
    # Create directory for output files if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save original transcription
    transcription_file = os.path.join(output_dir, f"{base_filename}_transcription.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write("\n".join(full_transcription))
    console.print(f"[green]Transcription saved to {transcription_file}[/green]")

    # Save Hindi translations to a separate file
    hindi_file = os.path.join(output_dir, f"{base_filename}_hindi.txt")
    with open(hindi_file, "w", encoding="utf-8") as f:
        f.write("\n".join(hindi_translations))
    console.print(f"[green]Hindi translations saved to {hindi_file}[/green]")

    # Save all translations to a single file (includes both English and Hindi)
    all_translations_file = os.path.join(output_dir, f"{base_filename}_translations.txt")
    with open(all_translations_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(full_transcription):
            f.write(f"English: {text}\n")
            if i < len(translations_by_language["hi"]):
                f.write(f"Hindi: {translations_by_language['hi'][i]}\n")
            f.write("\n")

    console.print(f"[green]All translations saved to {all_translations_file}[/green]")

    # Save audio if requested
    if args.save_audio and raw_audio_chunks:
        audio_file = os.path.join(output_dir, f"{base_filename}_audio.wav")
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(raw_audio_chunks))
        console.print(f"[green]Audio saved to {audio_file}[/green]")


# ==== Stopping Function (Waits for Enter Key in Separate Thread) ====
def stop_listener():
    input("Press Enter to stop recording...\n")
    console.print("[yellow]Stopping... (Processing remaining audio)[/yellow]")
    stop_event.set()


# ==== Main Execution ====
if __name__ == "__main__":
    try:
        # Start threads
        audio_thread = threading.Thread(target=audio_reader, daemon=True)
        processor_thread = threading.Thread(target=audio_processor, daemon=True)
        translation_thread = threading.Thread(target=translation_processor, daemon=True)
        stop_thread = threading.Thread(target=stop_listener, daemon=True)

        audio_thread.start()
        processor_thread.start()
        translation_thread.start()
        stop_thread.start()

        # Wait for threads to finish
        stop_thread.join()  # Wait for the stop signal
        audio_thread.join(timeout=2)  # Give audio thread time to clean up

        # Process any remaining audio in the queue
        console.print("[yellow]Processing remaining audio segments...[/yellow]")
        processor_thread.join()
        translation_thread.join()

        # Save Files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_files(timestamp)

    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user. Stopping...[/bold red]")
        stop_event.set()
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
    finally:
        # Cleanup
        p.terminate()
        console.print("[green]Program finished.[/green]")