# Video Translator with Voice Cloning and AI Voice Separation

**Author:** Mike Gazzaruso  
**License:** GNU/GPL v3  
**Version:** 0.6.0

## Overview
This project is a video translation pipeline that extracts speech from a video, transcribes it, translates it, and generates a voice-cloned speech using AI. The generated speech is then overlaid on the original video, replacing the original voice while preserving background sounds. It utilizes:

- **Whisper** for speech-to-text transcription
- **MBart** for machine translation
- **Tortoise-TTS** for voice cloning and text-to-speech synthesis
- **Demucs** for separating vocals and background audio
- **FFmpeg** for audio and video processing
- **Cache System** for reusing previously generated voice latents and models
- **Apple Silicon Support** for hardware acceleration on M1/M2/M3 Macs
- **CUDA Support** for hardware acceleration on Windows and Linux with NVIDIA GPUs

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
- **Real-time Process Control**: Ability to stop the translation process at any time
- **Live Progress Updates**: Detailed text display of process phases in real-time
- **Enhanced Error Handling**: Robust handling of file permissions and language codes
- **Broad format support**: Works with various video formats including iPhone videos (MOV), MP4, and others
- **English User Interface**: Complete English UI for improved accessibility and user experience
- **Consistent Default Voice**: Option to select male or female default voice when not using voice cloning
- **Hardware Acceleration**:
  - Apple Silicon (MPS) acceleration on M1/M2/M3 Macs
  - CUDA acceleration on Windows and Linux with NVIDIA GPUs
  - CPU fallback on systems without GPU acceleration (significantly slower, not recommended for production use)

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- FFmpeg (available via `apt`, `brew`, or `choco` depending on your OS)
- CUDA drivers (if running on NVIDIA GPU, optional but recommended)

### Install dependencies
Create a virtual environment and install required dependencies:
```bash
python -m venv video_translate_env
source video_translate_env/bin/activate  # On Windows use: video_translate_env\Scripts\activate
pip install -r requirements.txt
```

### Platform-specific Setup Scripts
We provide specialized setup scripts for different platforms that configure the environment for optimal performance:

#### Windows with NVIDIA GPU
```bash
# Run the Windows setup script
setup_windows.bat
```

#### Linux with NVIDIA GPU
```bash
# Make the script executable
chmod +x setup_linux.sh

# Run the setup script
./setup_linux.sh
```

#### macOS with Apple Silicon (M1/M2/M3)
```bash
# Make the script executable
chmod +x setup_mac.sh

# Run the setup script
./setup_mac.sh
```

These scripts will:
1. Check your platform and hardware
2. Create and activate a virtual environment
3. Install the correct version of PyTorch with appropriate acceleration (CUDA or MPS)
4. Install all other dependencies
5. Verify the installation

### Fixing CUDA Detection Issues
If you have an NVIDIA GPU but CUDA is not being detected, you can use our fix scripts:

#### Windows
```bash
# Run the CUDA fix script
fix_cuda.bat
```

#### Linux
```bash
# Make the script executable
chmod +x fix_cuda.sh

# Run the CUDA fix script
./fix_cuda.sh
```

### Verifying Hardware Acceleration
To verify hardware acceleration is working correctly:

#### For CUDA (Windows/Linux)
```bash
# Run the CUDA test script
python test_cuda.py
```

#### For MPS (macOS)
```bash
# Run the MPS check script
python check_mps.py

# Test Tortoise-TTS with MPS
python test_tortoise_mps.py
```

### Performance Considerations
**Important Note**: This application requires hardware acceleration for reasonable performance:

- **With GPU Acceleration** (CUDA or MPS): Voice synthesis typically takes 10-30 seconds per segment.
- **Without GPU Acceleration** (CPU only): Voice synthesis can take 5-10 minutes per segment, making the application impractically slow for most use cases.

If no accelerated hardware is detected, the application will automatically fall back to CPU processing, but this is not recommended for production use. For acceptable performance, we strongly recommend:

- An NVIDIA GPU with CUDA support (Windows/Linux)
- Apple Silicon Mac with MPS support (macOS)

### Streamlit GUI
The project includes a Streamlit-based graphical user interface for easier use:

```bash
# Run the Streamlit interface
streamlit run app.py

# Or use the convenience scripts
./run_streamlit.sh    # On Linux/Mac
run_streamlit.bat     # On Windows
```

#### Using the Streamlit Interface

The interface is divided into two main sections:

##### Sidebar

In the sidebar, you can configure:

- **Languages**: Select the source and target language for translation.
- **Cache options**: Enable/disable cache usage and clear existing cache.
- **Advanced options**: Configure advanced parameters for audio-video synchronization.
- **Temporary files**: Choose whether to keep temporary files after processing.

##### Main Panel

The main panel allows you to:

1. **Upload a video file**: Select the video you want to translate.
2. **Upload voice samples**: Provide audio samples for voice cloning.
3. **Select default voice**: Choose between male or female default voice when no voice samples are provided.
4. **Start the translation process**: Begin the translation with the configured settings.
5. **Monitor progress**: View real-time updates of the translation process.
6. **Download the result**: Get the translated video when processing is complete.

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
1. Upload the autodub.ipynb notebook to Google Colab
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
- `--use-female-voice` : Use a female voice as default when no voice samples are provided

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
├── device_utils.py         # Device selection utilities
├── tortoise_patch.py       # MPS compatibility patches
├── generate_requirements.py # Platform-specific requirements
├── setup_mac.sh            # macOS setup script
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

## Apple Silicon Acceleration

### Performance Expectations

MPS acceleration on Apple Silicon can provide significant speedups compared to CPU, but it may not be as fast as NVIDIA CUDA acceleration. Typical speedups range from 2x to 5x depending on the model and operations.

For Tortoise-TTS specifically, you can expect:
- Faster inference times
- Lower CPU usage
- Potentially higher memory usage

### Troubleshooting MPS Acceleration

If you're experiencing issues with MPS acceleration:

1. **Check PyTorch version**:
   MPS support requires PyTorch 1.12 or newer. Check your version with:
   ```python
   import torch
   print(torch.__version__)
   ```

2. **Verify MPS availability**:
   ```python
   import torch
   print(torch.backends.mps.is_available())
   print(torch.backends.mps.is_built())
   ```
   Both should return `True`.

3. **Reinstall PyTorch**:
   If MPS is not available, try reinstalling PyTorch:
   ```bash
   pip uninstall -y torch torchaudio
   pip install --upgrade torch torchaudio
   ```

4. **Known limitations**:
   - Not all PyTorch operations are supported by MPS yet
   - Some operations might be slower on MPS than on CPU
   - For very large models, you might encounter memory limitations

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

---

## License
This project is licensed under the **GNU General Public License v3.0**. You are free to modify and distribute it under the same terms.

---

## Credits
- **Mike Gazzaruso** - Developer & Creator
- Open-source AI models from OpenAI, Hugging Face, and community contributors.

For inquiries or contributions, feel free to open an issue or a pull request on the project repository.

