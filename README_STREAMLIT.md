# AutoDub - Streamlit Interface

This is a Streamlit-based graphical interface for the AutoDub project, which allows automatic video translation from one language to another while preserving the original voice through voice cloning and synchronizing the audio with the video.

## Installation

1. Make sure you have installed all project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

## Using the Interface

The interface is divided into two main sections:

### Sidebar

In the sidebar, you can configure:

- **Languages**: Select the source and target language for translation.
- **Cache options**: Enable/disable cache usage and clear existing cache.
- **Advanced options**: Configure advanced parameters for audio-video synchronization.
- **Temporary files**: Choose whether to keep temporary files after processing.

### Main Area

The main area is organized into two tabs:

#### Video Translation

In this tab, you can:

1. **Upload a video** to be translated.
2. **Upload audio samples** of the voice to be cloned (optional).
3. **Start the translation process**.
4. **Stop the translation process** at any time.
5. **View real-time details** of the translation process.
6. **Preview and download** the translated video.

#### Information

This tab contains **project information** and **instructions** on how to use the application.

## Workflow

1. **Configure the options** in the sidebar.
2. **Upload the video** to be translated.
3. **Upload voice samples** (optional).
4. **Start the translation** using the **"Start Translation"** button.
5. **Monitor progress** through real-time logs.
6. **Stop the process** at any time if needed using the **"Stop Translation"** button.
7. **Preview and download** the translated video once the process is complete.

## Important Notes

- The translation process **may take several minutes**, depending on the video length.
- The **quality of voice cloning** depends on the quality and quantity of the provided voice samples.
- For **optimal results**, provide **clear voice samples** without background noise.
- The application creates a **`conversions`** directory in the project folder to store translated videos.

## System Requirements

- **Python 3.8 or later**
- **GPU recommended** for better performance (especially for voice cloning)
- At least **8GB of RAM**
- **Sufficient disk space** for temporary files and translated videos
