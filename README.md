# Neural-Translator
# SPEECH_TRANSLATOR.IO 🎙️🌍  

Multilingual Speech Translation with AssemblyAI, ElevenLabs & Gradio

SPEECH_TRANSLATOR.IO is a simple web app that lets you:

- Record or upload **English audio**
- Automatically **transcribe** it using **AssemblyAI**
- **Translate** the text into multiple languages
- Generate **natural-sounding speech** using **ElevenLabs**
- Play or download the translated audio for each language

Currently supported target languages:

- 🇷🇺 Russian (`ru`)
- 🇹🇷 Turkish (`tr`)
- 🇩🇪 German (`de`)
- 🇪🇸 Spanish (`es`)
- 🇯🇵 Japanese (`ja`)
- 🇮🇳 Hindi (`hi`)

The UI is built with **Gradio** and runs entirely in the browser once the backend is started.

---

## ✨ Features

- 🎧 **English Speech-to-Text**  
  Uses **AssemblyAI** to transcribe uploaded or recorded English audio.

- 🌐 **Multi-language Translation**  
  Translates English transcripts into Russian, Turkish, German, Spanish, Japanese, and Hindi.

- 🗣️ **High-quality Text-to-Speech**  
  Uses **ElevenLabs** for natural multilingual speech synthesis.

- ⚡ **Parallel Processing**  
  Translations and TTS for all languages are processed **in parallel** using `ThreadPoolExecutor` for faster response.

- 🧠 **Result Caching**  
  - Transcripts are cached using an **MD5 hash** of the audio file.  
  - Translations are cached using a simple key `(text + target_language)`.

- 🧹 **Automatic Cleanup**  
  Temporary audio files are stored in a temp directory and cleaned up if older than 1 hour.

- 🖥️ **User-friendly Web UI**  
  Built with **Gradio Tabs** – each language has its own tab with:
  - Audio player (translated audio)
  - Translated text box

---

## 🧱 Tech Stack

- **Language:** Python 3.9+  
- **Libraries:**
  - [`gradio`](https://www.gradio.app/) – Web UI
  - [`assemblyai`](https://www.assemblyai.com/) – Speech-to-Text
  - [`translate`](https://pypi.org/project/translate/) – Text translation
  - [`elevenlabs`](https://elevenlabs.io/) – Text-to-Speech
  - Standard libraries: `os`, `uuid`, `time`, `tempfile`, `hashlib`, `concurrent.futures`, `functools`

---

## 🔐 Environment Variables

You must provide valid API keys via environment variables (recommended) **instead of hardcoding them**.

Set the following environment variables:

- `ASSEMBLYAI_API_KEY` – Your AssemblyAI API key  
- `ELEVENLABS_API_KEY` – Your ElevenLabs API key  
- `ELEVENLABS_VOICE_ID` – Voice ID from ElevenLabs (supports multilingual TTS)

Example (Linux / macOS):

```bash
export ASSEMBLYAI_API_KEY="your_assemblyai_key"
export ELEVENLABS_API_KEY="your_elevenlabs_key"
export ELEVENLABS_VOICE_ID="your_voice_id"
