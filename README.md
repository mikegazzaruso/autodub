# Video Translator with Voice Cloning and AI Voice Separation

**Author:** Mike Gazzaruso  
**License:** GNU/GPL v3  
**Version:** 0.3.1

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
- **Modular architecture for easy maintenance and extension**
- **Advanced Synchronization**: Aligns translated audio with the original video timing using:
  - Natural pause detection
  - Adaptive speed adjustment
  - Language-specific timing parameters
  - Intelligent segment splitting
- **Jupyter Notebook Support**: Easy integration with Google Colab via example.ipynb
- **Comprehensive Synchronization Metrics**: Detailed analysis of synchronization quality

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
pip install -r requirements.txt
```

---

## Usage
### Run Translation on a Video
To process a video with voice cloning:
```bash
python autodub.py --input path/to/video.mp4 --output path/to/output.mp4 --source-lang it --target-lang en --voice-samples path/to/voice_samples
```

### With Synchronization Options
```bash
python autodub.py --input path/to/video.mp4 --output path/to/output.mp4 --source-lang it --target-lang en --max-speed 1.5 --min-speed 0.8 --pause-threshold -30 --min-pause-duration 200
```

### Using a Synchronization Configuration File
Create a JSON file with your synchronization settings:
```json
{
  "max_speed_factor": 1.5,
  "min_speed_factor": 0.8,
  "pause_threshold": -30,
  "min_pause_duration": 200,
  "adaptive_timing": true,
  "preserve_sentence_breaks": true
}
```
Then run:
```bash
python autodub.py --input path/to/video.mp4 --output path/to/output.mp4 --source-lang it --target-lang en --sync-config sync_settings.json
```

### Using Google Colab
You can also use our Jupyter notebook for easy integration with Google Colab:
1. Upload the example.ipynb notebook to Google Colab
2. Follow the step-by-step instructions in the notebook
3. Upload your video and voice samples
4. Configure synchronization settings
5. Run the translation process
6. Download the translated video

### Options
- `--input` : Path to the input video file
- `--output` : Path to save the translated video
- `--source-lang` : Source language (e.g., `it` for Italian)
- `--target-lang` : Target language (e.g., `en` for English)
- `--voice-samples` : Directory containing `.wav` files for voice cloning
- `--no-cache` : Disable caching
- `--clear-cache` : Clear all cached data
- `--clear-voice-cache` : Clear only voice cache
- `--keep-temp` : Keep temporary files after processing
- `--sync-config` : Path to JSON file with synchronization configuration
- `--max-speed` : Maximum speed factor for audio adjustment
- `--min-speed` : Minimum speed factor for audio adjustment
- `--pause-threshold` : dB threshold for pause detection
- `--min-pause-duration` : Minimum pause duration in milliseconds
- `--no-adaptive-timing` : Disable adaptive timing based on language
- `--no-preserve-breaks` : Do not preserve sentence breaks

---

## Project Structure

```
video_translator/
├── __init__.py             # Package initialization
├── autodub.py              # Main entry point
├── video_translator.py     # VideoTranslator class
├── speech_recognition.py   # Speech recognition module
├── translation.py          # Translation module
├── voice_synthesis.py      # Voice synthesis module
├── audio_processing.py     # Audio processing module
├── sync_evaluation.py      # Synchronization evaluation module
├── utils.py                # Utility functions
└── autodub.ipynb           # Jupyter notebook for Google Colab
```

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
python autodub.py --clear-cache
```
To clear only the voice cache:
```bash
python autodub.py --clear-voice-cache
```

---

## Extending the Project
The modular architecture makes it easy to extend the project:

- **Add new languages**: Update the language map in `utils.py`
- **Improve voice synthesis**: Modify the voice synthesis module
- **Change transcription model**: Update the speech recognition module
- **Add new features**: Create new modules and integrate them into the workflow

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

