import os
import uuid
import time
import tempfile
import hashlib
import concurrent.futures
from functools import lru_cache

import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

# Load API keys from environment variables (best practice)
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "your key")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your key")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "your key")

# Create temporary directory for audio files
TEMP_DIR = tempfile.mkdtemp()
print(f"Using temporary directory: {TEMP_DIR}")

# Cache for transcripts and translations
TRANSCRIPT_CACHE = {}
TRANSLATION_CACHE = {}

# Define supported languages - Added Hindi
LANGUAGES = [
    {"code": "ru", "name": "Russian"},
    {"code": "tr", "name": "Turkish"},
    {"code": "de", "name": "German"},
    {"code": "es", "name": "Spanish"},
    {"code": "ja", "name": "Japanese"},
    {"code": "hi", "name": "Hindi"}  # Added Hindi only
]


# Function to get a unique hash for a file
def get_file_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        print(f"Error hashing file: {str(e)}")
        return None


# Function to transcribe audio using AssemblyAI with caching
def transcribe_audio(audio_file):
    try:
        if audio_file is None:
            return "No audio provided"

        # Check cache first
        file_hash = get_file_hash(audio_file)
        if file_hash and file_hash in TRANSCRIPT_CACHE:
            print("Using cached transcript")
            return TRANSCRIPT_CACHE[file_hash]

        aai.settings.api_key = ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_file)

        if transcript.status == aai.TranscriptStatus.error:
            return f"Transcription error: {transcript.error}"

        result = transcript.text or "No speech detected"

        # Cache the result
        if file_hash:
            TRANSCRIPT_CACHE[file_hash] = result

        return result
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return f"Error: {str(e)}"


# Function to translate text with caching
def translate_text(text, target_lang):
    try:
        # Generate cache key
        cache_key = f"{text[:100]}_{target_lang}"

        # Check cache first
        if cache_key in TRANSLATION_CACHE:
            print(f"Using cached translation for {target_lang}")
            return TRANSLATION_CACHE[cache_key]

        translator = Translator(from_lang="en", to_lang=target_lang)
        translation = translator.translate(text)

        # Cache the result
        TRANSLATION_CACHE[cache_key] = translation

        return translation
    except Exception as e:
        print(f"Translation error for {target_lang}: {str(e)}")
        return f"Translation to {target_lang} failed"


# Enhanced text-to-speech function
def text_to_speech(text, language_code):
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        # Create a unique filename in the temp directory
        filename = f"{language_code}_{uuid.uuid4()}.mp3"
        save_file_path = os.path.join(TEMP_DIR, filename)

        # Generate audio
        response = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
                style=0.5,
                use_speaker_boost=True,
            ),
        )

        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        # Verify file exists and has content
        if os.path.exists(save_file_path) and os.path.getsize(save_file_path) > 0:
            return save_file_path
        else:
            print(f"Warning: Audio file empty or not found: {save_file_path}")
            return None
    except Exception as e:
        print(f"Text-to-speech error for {language_code}: {str(e)}")
        return None


# Process a single language translation and TTS
def process_language(transcript_text, language_code, language_name):
    if not transcript_text or transcript_text.startswith("Error"):
        return None, f"Transcription failed: {transcript_text}"

    # Translate the text
    translated_text = translate_text(transcript_text, language_code)
    if not translated_text or translated_text.startswith("Translation to"):
        return None, f"Translation failed for {language_name}"

    # Generate speech from translated text
    audio_path = text_to_speech(translated_text, language_code)

    return audio_path, translated_text


# Process all languages in parallel
def process_all_languages(audio_file):
    # First transcribe the audio - this step must happen first
    transcript_text = transcribe_audio(audio_file)
    if not transcript_text or transcript_text.startswith("Error"):
        return transcript_text, None, None, None, None, None, None, None, None, None, None, None, None

    # Process translations in parallel
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(LANGUAGES)) as executor:
        future_to_lang = {}

        # Submit all language processing tasks
        for lang in LANGUAGES:
            future = executor.submit(process_language, transcript_text, lang["code"], lang["name"])
            future_to_lang[future] = lang["code"]

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_lang):
            lang_code = future_to_lang[future]
            try:
                audio_path, translated_text = future.result()
                results[lang_code] = (audio_path, translated_text)
            except Exception as e:
                print(f"Error processing {lang_code}: {str(e)}")
                results[lang_code] = (None, f"Error: {str(e)}")

    # Format results for the UI format - Swedish removed, Hindi added
    ru_audio, ru_text = results.get("ru", (None, "Translation failed"))
    tr_audio, tr_text = results.get("tr", (None, "Translation failed"))
    de_audio, de_text = results.get("de", (None, "Translation failed"))
    es_audio, es_text = results.get("es", (None, "Translation failed"))
    ja_audio, ja_text = results.get("ja", (None, "Translation failed"))
    hi_audio, hi_text = results.get("hi", (None, "Translation failed"))

    return (
        transcript_text,
        ru_audio, ru_text,
        tr_audio, tr_text,
        de_audio, de_text,
        es_audio, es_text,
        ja_audio, ja_text,
        hi_audio, hi_text
    )


# Clean up old temporary files
def cleanup_old_files():
    try:
        current_time = time.time()
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            # If file is older than 1 hour, delete it
            if os.path.isfile(file_path) and current_time - os.path.getmtime(file_path) > 3600:
                os.remove(file_path)
                print(f"Removed old file: {file_path}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


# Create Gradio interface with the modified tabbed UI
with gr.Blocks() as demo:
    gr.Markdown("# SPEECH_TRANSLATOR.IO")
    gr.Markdown("Record or upload audio in English and translate it to multiple languages.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Input Audio (English)",
            )

            # Display transcript
            transcript_output = gr.Textbox(label="English Transcript", interactive=False)

            with gr.Row():
                submit = gr.Button("Translate", variant="primary")
                clear_btn = gr.ClearButton([audio_input, transcript_output], value="Clear")

    # Show a loading message
    with gr.Row():
        status_msg = gr.Markdown("Record audio and click Translate to begin*")

    # Create tabs for different languages (Swedish removed, Hindi added)
    with gr.Tabs():
        with gr.TabItem("Russian"):
            ru_output = gr.Audio(label="Russian Audio", interactive=False)
            ru_text = gr.Textbox(label="Russian Text", interactive=False)

        with gr.TabItem("Turkish"):
            tr_output = gr.Audio(label="Turkish Audio", interactive=False)
            tr_text = gr.Textbox(label="Turkish Text", interactive=False)

        with gr.TabItem("German"):
            de_output = gr.Audio(label="German Audio", interactive=False)
            de_text = gr.Textbox(label="German Text", interactive=False)

        with gr.TabItem("Spanish"):
            es_output = gr.Audio(label="Spanish Audio", interactive=False)
            es_text = gr.Textbox(label="Spanish Text", interactive=False)

        with gr.TabItem("Japanese"):
            jp_output = gr.Audio(label="Japanese Audio", interactive=False)
            jp_text = gr.Textbox(label="Japanese Text", interactive=False)

        with gr.TabItem("Hindi"):
            hi_output = gr.Audio(label="Hindi Audio", interactive=False)
            hi_text = gr.Textbox(label="Hindi Text", interactive=False)

    # Process submit button click - using parallel processing with updated UI
    submit.click(
        fn=lambda: "Processing all translations in parallel... please wait",
        outputs=status_msg
    ).then(
        fn=process_all_languages,
        inputs=audio_input,
        outputs=[
            transcript_output,
            ru_output, ru_text,
            tr_output, tr_text,
            de_output, de_text,
            es_output, es_text,
            jp_output, jp_text,
            hi_output, hi_text
        ],
        queue=True
    ).then(
        fn=lambda: "Translations complete!",
        outputs=status_msg
    ).then(
        fn=cleanup_old_files
    )

    # Add instructions
    gr.Markdown("""
    ### How to use
    1. Click the microphone button to record or upload an audio file in English
    2. Click "Translate" to process
    3. View and play translations in the tabs below

    ### Troubleshooting
    - If you encounter errors, try a shorter audio clip
    - Make sure your internet connection is stable
    - Speak clearly for better transcription results
    - For Hindi audio, please note that ElevenLabs may have limited support for certain languages
    """)

if __name__ == "__main__":
    # Add a check for empty API keys
    if not ASSEMBLYAI_API_KEY or ASSEMBLYAI_API_KEY.startswith("<"):
        print("Warning: AssemblyAI API key is not properly set!")
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY.startswith("<"):
        print("Warning: ElevenLabs API key is not properly set!")

    # Launch with share option
    demo.launch()
