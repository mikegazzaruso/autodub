# Video Translator with Voice Cloning and AI Voice Separation

**Author:** Mike Gazzaruso  
**License:** GNU/GPL v3  

## Overview
This project is a video translation pipeline that extracts speech from a video, transcribes it, translates it, and generates a voice-cloned speech using AI. The generated speech is then overlaid on the original video, replacing the original voice while preserving background sounds. It utilizes:

- **Whisper** for speech-to-text transcription
- **MBart** for machine translation
- **Tortoise-TTS** for voice cloning and text-to-speech synthesis
- **Demucs** for separating vocals and background audio
- **FFmpeg** for audio and video processing
- **Cache System** for reusing previously generated voice latents and models

**Note:** Lip-sync support is planned for future releases.

---

## Features
- **Automated speech-to-text transcription using Whisper**
- **Machine translation of transcribed text using MBart**
- **Voice cloning and text-to-speech synthesis with Tortoise-TTS**
- **Separation of vocals and background music using Demucs**
- **Caching mechanism to speed up repeated voice cloning processes**
- **Full video processing pipeline with FFmpeg**

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- FFmpeg (available via `apt`, `brew`, or `choco` depending on your OS)
- CUDA (if running on GPU, optional but recommended)

### Install dependencies
Create a virtual environment and install required dependencies:
```bash
python -m venv video_translate_env
source video_translate_env/bin/activate  # On Windows use: video_translate_env\Scripts\activate
pip install torch torchaudio torchvision
pip install openai-whisper transformers librosa moviepy demucs soundfile pydub ffmpeg-python tqdm numpy scipy
```

---

## Usage
### Run Translation on a Video
To process a video with voice cloning:
```bash
python video_translator.py --input path/to/video.mp4 --output path/to/output.mp4 --source-lang it --target-lang en --voice-samples path/to/voice_samples
```

### Options
- `--input` : Path to the input video file
- `--output` : Path to save the translated video
- `--source-lang` : Source language (e.g., `it` for Italian)
- `--target-lang` : Target language (e.g., `en` for English)
- `--voice-samples` : Directory containing `.wav` files for voice cloning
- `--no-cache` : Disable caching
- `--clear-cache` : Clear all cached data
- `--clear-voice-cache` : Clear only voice cache

---

## Pipeline Workflow
1. **Extract audio**: The script extracts the original audio from the video.
2. **Transcription**: Whisper transcribes the speech.
3. **Translation**: MBart translates the text into the target language.
4. **Voice Cloning & Speech Synthesis**: Tortoise-TTS generates new audio with a cloned voice.
5. **AI Voice Separation**: Demucs separates voice and background sounds.
6. **Merge Translated Audio**: The new translated voice is combined with background audio.
7. **Reintegrate Audio & Video**: The final audio is merged with the original video.

---

## Caching System
This project implements a caching mechanism to speed up repeated processing:
- **Whisper model caching**: The speech-to-text model is stored to avoid reloading.
- **Voice conditioning latents caching**: Tortoise-TTS voice latents are stored to prevent redundant computation.
- **Preprocessed voice samples caching**: Speeds up voice cloning across multiple runs.

To clear cached models and latents:
```bash
python video_translator.py --clear-cache
```
To clear only the voice cache:
```bash
python video_translator.py --clear-voice-cache
```

---

## Future Improvements
- **Lip Sync Support**: The system will soon include precise lip synchronization.
- **GUI Interface**: A graphical user interface will be added for ease of use.
- **More Language Support**: Additional languages will be supported in future updates.

---

## License
This project is licensed under the **GNU General Public License v3.0**. You are free to modify and distribute it under the same terms.

---

## Credits
- **Mike Gazzaruso** - Developer & Creator
- Open-source AI models from OpenAI, Hugging Face, and community contributors.

For inquiries or contributions, feel free to open an issue or a pull request on the project repository.

